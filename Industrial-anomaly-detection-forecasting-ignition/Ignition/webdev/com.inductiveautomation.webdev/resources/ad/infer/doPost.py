def doPost(request, session):

   # Beklenmeyen hata durumunda detaylı traceback almak için kullanılır.
   import traceback
   try:
       # system: Ignition fonksiyonlarına erişim sağlar.
       # json: request/response verilerini JSON formatında encode/decode etmek için kullanılır.
       # anomaly: anomaly.py içindeki inferTags fonksiyonunu çağırmak için kullanılır.
       import system, json, anomaly
       # Bu endpointin loglarını Gateway Logs altında WEBDEV-INFER etiketiyle gösterir.
       log = system.util.getLogger("WEBDEV-INFER")
       # Endpointin çalışmaya başladığını loglar.
       log.info("WEBDEV INFER STARTED")

       # GÜVENLİ FLOAT DÖNÜŞÜMÜ
       # None, boş veya hatalı değer gelirse default float değer döndürür.
       def safe_float(x, default=0.0):
           try:
               # Değer None ise default döndürülür.
               if x is None:
                   return default
               # Değer float'a çevrilir.
               return float(x)
           except:
               # Çevrim hatasında default döndürülür.
               return default

       # GÜVENLİ INT DÖNÜŞÜMÜ
       # None, boş veya hatalı değer gelirse default int değer döndürür.
       def safe_int(x, default=0):
           try:
               # Değer None ise default döndürülür.
               if x is None:
                   return default
               # Değer int'e çevrilir.
               return int(x)
           except:
               # Çevrim hatasında default döndürülür.
               return default

       # GÜVENLİ TAG WRITE
       # Tag write işlemi başarısız olsa bile endpointin tamamen crash olmasını engeller.
       def safe_write(paths, values, context="TAG WRITE"):
           try:
               # Verilen tag path listesine karşılık gelen değer listesi yazılır.
               system.tag.writeBlocking(paths, values)
               # Başarılı yazma işlemi loglanır.
               log.info("%s OK | count=%s" % (context, len(paths)))
           except Exception as tag_e:
               # Tag write hatası loglanır.
               log.error("%s FAILED | %s" % (context, tag_e))
               
       # JSON-SAFE DÖNÜŞÜM
       # Ignition/Jython içinde response objesini güvenli JSON dict formatına çevirmek için kullanılır.
       def to_json_safe(obj):
           try:
               # Nesne önce JSON string'e encode edilir, sonra tekrar decode edilir.
               return system.util.jsonDecode(system.util.jsonEncode(obj))
           except:
               # Dönüşüm başarısız olursa obje olduğu gibi döndürülür.
               return obj
               
       # HATA RESPONSE HELPER
       # Hata durumunda hem ml taglerini günceller hem de kullanıcıya JSON response döndürür.
       def return_error_response(msg, http_status, payload_obj, infer_obj, latency_ms):
           # Infer response JSON-safe hale getirilir.
           infer_safe_local = to_json_safe(infer_obj)
           # Hata bilgileri ml/evidence taglerine yazılır.
           safe_write(
               [
                   "[default]ml/Backend_Status",
                   "[default]ml/Backend_ErrorMessage",
                   "[default]ml/HttpStatusCode",
                   "[default]ml/LastLatencyMs",
                   "[default]ml/LastPrediction",
                   "[default]ml/LastPredictionText",
                   "[default]ml/LastRequestJson",
                   "[default]ml/LastResponseJson",
                   "[default]ml/LastResponseTime",
                   "[default]evidence/BackendReceivedData"
               ],
               [
                   "ERROR",
                   msg,
                   int(http_status),
                   int(latency_ms),
                   False,
                   "NORMAL",
                   system.util.jsonEncode(payload_obj),
                   system.util.jsonEncode(infer_safe_local),
                   system.date.now(),
                   False
               ],
               "ML ERROR TAG WRITE"
           )
           # WebDev response JSON formatında döndürülür.
           return {
           "json":{
           "ok":False,
           "user_message":msg,
           "status":int(http_status),
           "error":{
           "infer":infer_obj
           }
           }
           }

       # INFER RESPONSE ÖZETİ ÇIKARMA
       # Backend /infer response içinden toplam anomali, event anomali, maksimum score ve threshold değerlerini çıkarır.
       def extract_infer_summary(infer_resp):
           # Nokta bazlı toplam anomali sayısı.
           total_anoms = 0
           # Event bazlı toplam anomali sayısı.
           total_event_anoms = 0
           # En yüksek anomaly score değeri.
           score = 0.0
           # En yüksek threshold değeri.
           threshold = 0.0
           # infer_resp dict değilse boş özet döndürülür.
           if not isinstance(infer_resp, dict):
               return total_anoms, total_event_anoms, score, threshold
           # Backend response içinden results listesi alınır.
           results = infer_resp.get("results", [])
           # results boşsa boş özet döndürülür.
           if not results:
               return total_anoms, total_event_anoms, score, threshold
           # Her tag sonucu gezilir.
           for r in results:
               # Result dict değilse atlanır.
               if not isinstance(r, dict):
                   continue
               # Tag seviyesindeki summary alınır.
               top_summary = r.get("summary", {})
               # Summary dict ise top-level toplamlar okunur.
               if isinstance(top_summary, dict):
                   # Backend total_anomaly_count döndürüyorsa toplam anomali sayısına eklenir.
                   total_anoms += safe_int(top_summary.get("total_anomaly_count"), 0)
                   # Backend total_anomaly_event_count döndürüyorsa event sayısına eklenir.
                   total_event_anoms += safe_int(top_summary.get("total_anomaly_event_count"), 0)
                   # Threshold farklı field isimleriyle gelebileceği için adaylar listelenir.
                   top_thr_candidates = [
                       top_summary.get("threshold_used"),
                       top_summary.get("threshold"),
                       top_summary.get("score_threshold")
                   ]
                   # Threshold adayları gezilir.
                   for c in top_thr_candidates:
                       # Aday float'a çevrilir.
                       cf = safe_float(c, 0.0)
                       # En büyük threshold değeri tutulur.
                       if cf > threshold:
                           threshold = cf
               # Batch listesi alınır.
               batches = r.get("batches", [])
               # Her batch ayrı ayrı gezilir.
               for b in batches:
                   # Batch dict değilse atlanır.
                   if not isinstance(b, dict):
                       continue
                   # Batch summary alınır.
                   summary = b.get("summary", {})
                   # Summary dict ise batch-level sayılar ve threshold okunur.
                   if isinstance(summary, dict):
                       # Top-level toplam yoksa batch anomaly_count üzerinden toplam hesaplanır.
                       if safe_int(top_summary.get("total_anomaly_count"), 0) == 0:
                           total_anoms += safe_int(summary.get("anomaly_count"), 0)
                       # Top-level event toplamı yoksa batch anomaly_event_count üzerinden toplam hesaplanır.
                       if safe_int(top_summary.get("total_anomaly_event_count"), 0) == 0:
                           total_event_anoms += safe_int(summary.get("anomaly_event_count"), 0)
                       # Batch threshold adayları listelenir.
                       thr_candidates = [
                           summary.get("threshold_used"),
                           summary.get("threshold"),
                           summary.get("score_threshold")
                       ]
                       # Batch threshold adayları gezilir.
                       for c in thr_candidates:
                           # Aday float'a çevrilir.
                           cf = safe_float(c, 0.0)
                           # En büyük threshold değeri tutulur.
                           if cf > threshold:
                               threshold = cf
                   # Anomaly score zaman serisi alınır.
                   anomaly_scores = b.get("anomaly_score", [])
                   # Score listesi geçerliyse en yüksek value bulunur.
                   if isinstance(anomaly_scores, list):
                       for p in anomaly_scores:
                           if isinstance(p, dict):
                               v = safe_float(p.get("value"), 0.0)
                               if v > score:
                                   score = v
                   # Nokta bazlı anomalies listesi alınır.
                   anomalies = b.get("anomalies", [])
                   # Anomalies içindeki en yüksek error değeri score olarak da kontrol edilir.
                   if isinstance(anomalies, list):
                       for a in anomalies:
                           if isinstance(a, dict):
                               err = safe_float(a.get("error"), 0.0)
                               if err > score:
                                   score = err
                   # Threshold grafiği için gelen threshold_series alınır.
                   threshold_series = b.get("threshold_series", [])
                   # Threshold series içindeki value değerlerinden en büyüğü tutulur.
                   if isinstance(threshold_series, list):
                       for p in threshold_series:
                           if isinstance(p, dict):
                               v = safe_float(p.get("value"), 0.0)
                               if v > threshold:
                                   threshold = v
           # Çıkarılan özet değerleri döndürülür.
           return total_anoms, total_event_anoms, score, threshold

       # REQUEST PAYLOAD OKUMA
       # WebDev request içinden postData alınır.
       raw = request.get("postData")
       # Raw postData loglanır.
       log.info("RAW POSTDATA = %s" % raw)
       # postData tipi loglanır.
       log.info("POSTDATA TYPE = %s" % type(raw))
       # postData string ise JSON decode edilir.
       if isinstance(raw, basestring):
           payload = system.util.jsonDecode(raw)
       # postData zaten dict ise direkt kullanılır.
       elif isinstance(raw, dict):
           payload = raw
       # Beklenmeyen tip gelirse hata fırlatılır.
       else:
           raise ValueError("postData must be string or dict")
       # Decode edilmiş payload loglanır.
       log.info("DECODED PAYLOAD = %s" % payload)
       # Payload içinden tag path alınır.
       tag_path = payload.get("tagPath")
       # Payload içinden backend URL alınır.
       backend = payload.get("backend")
       # Inference için kaç dakikalık history çekileceği alınır.
       infer_minutes = safe_int(payload.get("inferMinutes"), 30)
       # Threshold/score_threshold değeri alınır; yoksa -0.5 default kullanılır.
       threshold = safe_float(payload.get("threshold", payload.get("score_threshold")), -0.5)
       # Holdout minutes alınır; yoksa infer_minutes veya default 10 kullanılır.
       holdout_minutes = safe_int(payload.get("holdoutMinutes", infer_minutes), 10)
       # Veri örnekleme aralığı saniye cinsinden alınır.
       step_seconds = safe_int(payload.get("stepSeconds"), 5)
       # Batch süresi dakika cinsinden alınır.
       batch_minutes = payload.get("batchMinutes", 10)
       # Batch stride verilmezse batch_minutes kullanılır.
       batch_stride_minutes = payload.get("batchStrideMinutes", batch_minutes)
       # Anomaly mode'a geçmek için kaç ardışık anomali gerektiği alınır.
       anomaly_confirm_points = safe_int(payload.get("anomaly_confirm_points"), 1)
       # Normal mode'a geri dönmek için kaç temiz nokta gerektiği alınır.
       recover_points = safe_int(payload.get("recover_points"), 6)
       # Normalize 0-1 aralığı dışına çıkma toleransı alınır.
       range_margin = safe_float(payload.get("range_margin"), 0.03)
       # Backend zorunlu olduğu için boşsa hata fırlatılır.
       if not backend:
           raise ValueError("backend is required")
       # Tag path zorunlu olduğu için boşsa hata fırlatılır.
       if not tag_path:
           raise ValueError("tagPath is required")
       # anomaly.py içindeki global BACKEND değişkeni UI'dan gelen backend ile güncellenir.
       anomaly.BACKEND = backend
       # Backend adresi loglanır.
       log.info("BACKEND SET = %s" % anomaly.BACKEND)
       # Seçili tag path loglanır.
       log.info("tagPath = %s" % tag_path)
       # Infer minutes loglanır.
       log.info("inferMinutes = %s" % infer_minutes)
       # Threshold loglanır.
       log.info("threshold = %s" % threshold)
       # Holdout minutes loglanır.
       log.info("holdoutMinutes = %s" % holdout_minutes)
       # Step seconds loglanır.
       log.info("stepSeconds = %s" % step_seconds)
       # Batch minutes loglanır.
       log.info("batchMinutes = %s" % batch_minutes)
       # Batch stride minutes loglanır.
       log.info("batchStrideMinutes = %s" % batch_stride_minutes)
       # Anomaly confirm points loglanır.
       log.info("anomaly_confirm_points = %s" % anomaly_confirm_points)
       # Recover points loglanır.
       log.info("recover_points = %s" % recover_points)
       # Range margin loglanır.
       log.info("range_margin = %s" % range_margin)
 
       # INFER ÇAĞRISI
       # Infer başlangıç zamanı milisaniye olarak alınır.
       infer_start_ms = system.date.now().getTime()
       # anomaly.py içindeki inferTags fonksiyonu çağrılır.
       infer_resp = anomaly.inferTags(
           [tag_path],
           minutes=infer_minutes,
           thr=threshold,
           holdout_minutes=holdout_minutes,
           step_seconds=step_seconds,
           batch_minutes=batch_minutes,
           batch_stride_minutes=batch_stride_minutes,
           anomaly_confirm_points=anomaly_confirm_points,
           recover_points=recover_points,
           range_margin=range_margin
       )
       # Infer işleminin kaç milisaniye sürdüğü hesaplanır.
       infer_latency_ms = int(system.date.now().getTime() - infer_start_ms)
       # Infer response tipi loglanır.
       log.info("TYPE INFER = %s" % type(infer_resp))
       # Infer response içeriği loglanır.
       log.info("INFER RESP = %s" % infer_resp)
       # Infer response JSON-safe hale getirilir.
       infer_safe = to_json_safe(infer_resp)

       # INFER HATA KONTROLÜ
       # infer_resp içinde error varsa hata akışı çalışır.
       if isinstance(infer_resp, dict) and infer_resp.get("error"):
           # Infer hatası loglanır.
           log.error("INFER failed: %s" % infer_resp)
           # Default hata mesajı.
           msg = "Infer process error"
           # Default HTTP status.
           http_status = 500
           # Backend 2xx dışı cevap verdiyse status ve detail backend response'tan alınır.
           if infer_resp.get("error") == "backend_non_2xx":
               http_status = int(infer_resp.get("status", 500))
               if infer_resp.get("detail"):
                   msg = "Infer failed: %s" % infer_resp.get("detail")
               else:
                   msg = "Infer failed HTTP %s" % http_status
           # Backend kapalı/network timeout ise kullanıcı dostu mesaj verilir.
           elif infer_resp.get("error") in ["network_timeout", "backend_unreachable"]:
               http_status = 503
               msg = infer_resp.get("detail") or "FastAPI server could not be reached."
           # Veri yetersizliği gibi diğer hata tiplerinde detail varsa kullanıcıya gösterilir.
           elif infer_resp.get("detail"):
               msg = infer_resp.get("detail")
           # Standart hata response'u döndürülür.
           return return_error_response(msg, http_status, payload, infer_safe, infer_latency_ms)
           
       # INFER ÖZETİ ÇIKARMA
       # Backend response içinden toplam anomali, event sayısı, en yüksek score ve threshold alınır.
       total_anoms, total_event_anoms, last_score, last_threshold = extract_infer_summary(infer_resp)
       # Eğer response içinde threshold bulunamazsa payload threshold değeri fallback olarak kullanılır.
       if last_threshold == 0.0:
           last_threshold = safe_float(payload.get("threshold"), 0.0)
       # Toplam anomali sayısı 0'dan büyükse cihaz anomali durumunda kabul edilir.
       actual_is_anomaly = total_anoms > 0
       # Kullanıcıya gösterilecek status text'i belirlenir.
       actual_status = "ANOMALY" if actual_is_anomaly else "NORMAL"

       # ML TAGLERİNİ GÜNCELLE
       # Yazılacak ml tag pathleri.
       ml_paths = [
           "[default]ml/Backend_Status",
           "[default]ml/Backend_ErrorMessage",
           "[default]ml/HttpStatusCode",
           "[default]ml/LastLatencyMs",
           "[default]ml/LastPrediction",
           "[default]ml/LastPredictionText",
           "[default]ml/LastRequestJson",
           "[default]ml/LastResponseJson",
           "[default]ml/LastResponseTime",
           "[default]ml/LastScore",
           "[default]ml/LastThreshold"
       ]
       # Yazılacak ml tag değerleri.
       ml_values = [
           "OK",
           "",
           200,
           int(infer_latency_ms),
           bool(actual_is_anomaly),
           actual_status,
           system.util.jsonEncode(payload),
           system.util.jsonEncode(infer_safe),
           system.date.now(),
           float(last_score),
           float(last_threshold)
       ]
       # ML tagleri güvenli şekilde güncellenir.
       safe_write(ml_paths, ml_values, "ML SUCCESS TAG WRITE")

       # SON TEST SONUCUNU GÜNCELLE
       # Test/evidence için okunacak tag pathleri.
       result_read_paths = [
           "[default]results/TestCaseName",
           "[default]results/Expected_IsAnomaly",
           "[default]results/Expected_Status",
           "[default]evidence/LastSensorName",
           "[default]evidence/LastInputValue"
       ]
       # Test/evidence tagleri okunur.
       result_read_values = system.tag.readBlocking(result_read_paths)
       # Test case adı okunur; kalite kötü veya değer yoksa NONE kabul edilir.
       test_case_name = str(result_read_values[0].value) if result_read_values[0].quality.isGood() and result_read_values[0].value is not None else "NONE"
       # Beklenen anomaly boolean değeri okunur.
       expected_is_anomaly = bool(result_read_values[1].value) if result_read_values[1].quality.isGood() else False
       # Beklenen status text'i okunur.
       expected_status = str(result_read_values[2].value) if result_read_values[2].quality.isGood() and result_read_values[2].value is not None else "NORMAL"
       # Son sensör adı okunur.
       last_sensor_name = str(result_read_values[3].value) if result_read_values[3].quality.isGood() and result_read_values[3].value is not None else "NONE"
       # Son input value güvenli float olarak okunur.
       last_input_value = safe_float(result_read_values[4].value, 0.0)
       # Eğer test case tanımlıysa test sonucu hesaplanır.
       if test_case_name != "NONE":
           # Dropout handled senaryosu özel olarak başarılı kabul edilir.
           if expected_status == "HANDLED_DROPOUT":
               test_passed = True
           # Diğer senaryolarda beklenen anomaly ile actual anomaly karşılaştırılır.
           else:
               test_passed = (expected_is_anomaly == actual_is_anomaly)
           # Test sonucu PASS/FAIL olarak hazırlanır.
           result_text = "PASS" if test_passed else "FAIL"
           # Test mesajı detaylı olarak hazırlanır.
           test_message = "Infer refresh completed. Sensor=%s, Input=%.4f, Expected=%s, Actual=%s, AnomalyCount=%s, EventCount=%s, Score=%.6f, Threshold=%.6f" % (
               last_sensor_name,
               last_input_value,
               expected_status,
               actual_status,
               total_anoms,
               total_event_anoms,
               last_score,
               last_threshold
           )
           # Yazılacak result/evidence tag pathleri.
           result_write_paths = [
               "[default]results/Actual_IsAnomaly",
               "[default]results/Actual_Status",
               "[default]results/TestPassed",
               "[default]results/TestResultText",
               "[default]results/TestMessage",
               "[default]results/LastTestTime",
               "[default]results/DetectionLatencyMs",
               "[default]evidence/BackendReceivedData"
           ]
           # Yazılacak result/evidence değerleri.
           result_write_values = [
               bool(actual_is_anomaly),
               actual_status,
               bool(test_passed),
               result_text,
               test_message,
               system.date.now(),
               int(infer_latency_ms),
               True
           ]
           # Test sonucu tagleri güvenli şekilde yazılır.
           safe_write(result_write_paths, result_write_values, "RESULT TAG WRITE")
       # Endpointin başarılı bittiği loglanır.
       log.info("WEBDEV INFER ENDED SUCCESSFULLY")
       # Perspective/UI tarafına başarılı JSON response döndürülür.
       return {
           "json": {
               "ok": True,
               "infer": infer_resp,
               "summary": {
                   "total_anomaly_count": total_anoms,
                   "total_anomaly_event_count": total_event_anoms,
                   "last_score": last_score,
                   "last_threshold": last_threshold,
                   "latency_ms": infer_latency_ms
               }
           }
       }

   # GLOBAL CRASH HANDLER
   # Beklenmeyen herhangi bir hata olursa burası çalışır.
   except Exception as e:
       # Detaylı traceback alınır.
       err_msg = traceback.format_exc()
       # Logger tekrar alınır; try içinde log oluşmadan hata olduysa garanti eder.
       log = system.util.getLogger("WEBDEV-INFER")
       # Crash detayları loglanır.
       log.error("CRASH: " + err_msg)
       # Hatalı bitiş loglanır.
       log.info("WEBDEV INFER ENDED WITH ERROR")
       # Crash durumunda ml/evidence tagleri hata durumuna çekilir.
       try:
           system.tag.writeBlocking(
               [
                   "[default]ml/Backend_Status",
                   "[default]ml/Backend_ErrorMessage",
                   "[default]ml/HttpStatusCode",
                   "[default]ml/LastPrediction",
                   "[default]ml/LastPredictionText",
                   "[default]ml/LastResponseTime",
                   "[default]evidence/BackendReceivedData"
               ],
               [
                   "ERROR",
                   str(e),
                   500,
                   False,
                   "NORMAL",
                   system.date.now(),
                   False
               ]
           )
       # Tag write bile başarısız olursa endpoint yine response döndürsün diye pass edilir.
       except:
           pass
       # UI tarafına crash bilgisi JSON olarak döndürülür.
       return {
           "json": {
               "ok": False,
               "error_message": str(e),
               "traceback": err_msg
           }
       }