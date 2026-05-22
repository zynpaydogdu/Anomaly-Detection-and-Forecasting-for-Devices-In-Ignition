def doPost(request, session):

    # Hata olduğunda detaylı traceback almak için kullanılır.
    import traceback
    try:
        # system: Ignition fonksiyonlarına erişim sağlar.
        # json: payload encode/decode için kullanılır.
        # anomaly: anomaly.py scriptindeki trainTags ve inferTags fonksiyonlarını çağırmak için kullanılır.
        import system, json, anomaly
        # Bu WebDev endpointine ait logları Gateway Logs altında WEBDEV-TRAIN etiketiyle gösterir.
        log = system.util.getLogger("WEBDEV-TRAIN")
        # Endpointin çalışmaya başladığını loglar.
        log.info("WEBDEV TRAIN STARTED")

        # GÜVENLİ FLOAT DÖNÜŞÜMÜ
        # None, boş veya hatalı değer gelirse default değer döndürür.
        def safe_float(x, default=0.0):
            try:
                # Değer None ise default dön.
                if x is None:
                    return default
                # Değeri float'a çevirmeyi dene.
                return float(x)
            except:
                # Çevrilemezse default dön.
                return default

        # GÜVENLİ INT DÖNÜŞÜMÜ
        # None, boş veya hatalı değer gelirse default int değer döndürür.
        def safe_int(x, default=0):
            try:
                # Değer None ise default dön.
                if x is None:
                    return default
                # Değeri int'e çevirmeyi dene.
                return int(x)
            except:
                # Çevrilemezse default dön.
                return default
        # IGNITION TAG WRITE HELPER
        # Tag yazma işlemini try/except içine alır; tag yazılamazsa endpointin tamamen düşmesini engeller.
        def safe_write(paths, values, context="TAG WRITE"):
            try:
                # Verilen path listesine karşılık gelen value listesi yazılır.
                system.tag.writeBlocking(paths, values)
                # Başarılı tag write loglanır.
                log.info("%s OK | count=%s" % (context, len(paths)))
            except Exception as tag_e:
                # Tag write hatası loglanır ama ana akış kırılmaz.
                log.error("%s FAILED | %s" % (context, tag_e))

        # RESPONSE JSON SAFE HELPER
        # Ignition/Jython içinde Java/Python nesnelerini JSON-safe dict'e çevirmek için kullanılır.
        def to_json_safe(obj):
            try:
                # Önce JSON string'e encode edip tekrar decode ederek safe hale getirir.
                return system.util.jsonDecode(system.util.jsonEncode(obj))
            except:
                # Encode/decode başarısız olursa obje olduğu gibi döner.
                return obj

        # HATA RESPONSE HELPER
        # Hem WebDev response döndürür hem ml/evidence taglerini hata durumuna çeker.
        def return_error_response(msg, http_status, payload_obj, train_obj, infer_obj, latency_ms):
            # Train response JSON-safe hale getirilir.
            train_safe_local = to_json_safe(train_obj)
            # Infer response JSON-safe hale getirilir.
            infer_safe_local = to_json_safe(infer_obj)
            # Backend hata bilgisini ml ve evidence taglerine yazar.
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
                    system.util.jsonEncode({
                        "train": train_safe_local,
                        "infer": infer_safe_local
                    }),
                    system.date.now(),
                    False
                ],
                "ML ERROR TAG WRITE"
            )
            # WebDev tarafına JSON response döndürür.
            return {
           "json":{
           "ok":False,
           "user_message":msg,
           "status":500,
           "error":{
           "infer":infer_obj
           }
           }
           }

        # INFER RESPONSE ÖZETİ ÇIKARMA
        # Backend /infer response içinden toplam anomali sayısı, en yüksek score ve threshold değerini çıkarır.
        def extract_infer_summary(infer_resp):
            # Toplam anomali sayısı başlangıç değeri.
            total_anoms = 0
            # En yüksek anomaly score başlangıç değeri.
            score = 0.0
            # En yüksek threshold başlangıç değeri.
            threshold = 0.0
            # infer_resp dict değilse boş sonuç döndür.
            if not isinstance(infer_resp, dict):
                return total_anoms, score, threshold
            # Backend response içindeki results listesi alınır.
            results = infer_resp.get("results", [])
            # results boşsa boş sonuç döndür.
            if not results:
                return total_anoms, score, threshold
            # Her tag sonucu gezilir.
            for r in results:
                # Result dict değilse atla.
                if not isinstance(r, dict):
                    continue
                # Tag seviyesindeki summary alınır.
                top_summary = r.get("summary", {})
                # Tag summary dict ise toplam anomali ve threshold bilgileri okunur.
                if isinstance(top_summary, dict):
                    # Backend total_anomaly_count döndürüyorsa doğrudan toplam sayıya eklenir.
                    total_anoms += safe_int(top_summary.get("total_anomaly_count"), 0)
                    # Threshold için muhtemel field isimleri listelenir.
                    for c in [
                        top_summary.get("threshold_used"),
                        top_summary.get("threshold"),
                        top_summary.get("score_threshold")
                    ]:
                        # Threshold adayı float'a çevrilir.
                        cf = safe_float(c, 0.0)
                        # En büyük threshold değeri tutulur.
                        if cf > threshold:
                            threshold = cf
                # Batch sonuçları alınır.
                batches = r.get("batches", [])
                # Eğer batch varsa batch içi anomaly_score/anomalies/threshold bilgileri okunur.
                for b in batches:
                    # Batch dict değilse atla.
                    if not isinstance(b, dict):
                        continue
                    # Batch summary alınır.
                    summary = b.get("summary", {})
                    # Batch summary dict ise anomaly_count ve threshold okunur.
                    if isinstance(summary, dict):
                        # Eğer top level total yoksa batch anomaly_count değerlerinden toplam hesaplanır.
                        if safe_int(top_summary.get("total_anomaly_count"), 0) == 0:
                            total_anoms += safe_int(summary.get("anomaly_count"), 0)
                        # Batch threshold adayları gezilir.
                        for c in [
                            summary.get("threshold_used"),
                            summary.get("threshold"),
                            summary.get("score_threshold")
                        ]:
                            # Threshold adayı float'a çevrilir.
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
                    # Anomalies içinden en yüksek error bulunur.
                    if isinstance(anomalies, list):
                        for a in anomalies:
                            if isinstance(a, dict):
                                err = safe_float(a.get("error"), 0.0)
                                if err > score:
                                    score = err
                    # Threshold çizgi serisi alınır.
                    threshold_series = b.get("threshold_series", [])
                    # Threshold series içinden en yüksek value bulunur.
                    if isinstance(threshold_series, list):
                        for p in threshold_series:
                            if isinstance(p, dict):
                                v = safe_float(p.get("value"), 0.0)
                                if v > threshold:
                                    threshold = v
                # Eğer response batch içermiyorsa eski/flat response yapısı desteklenir.
                if not batches:
                    # Result summary alınır.
                    summary = r.get("summary", {})
                    # Summary dict ise anomaly_count ve threshold okunur.
                    if isinstance(summary, dict):
                        total_anoms += safe_int(summary.get("anomaly_count"), 0)
                        for c in [
                            summary.get("threshold_used"),
                            summary.get("threshold"),
                            summary.get("score_threshold")
                        ]:
                            cf = safe_float(c, 0.0)
                            if cf > threshold:
                                threshold = cf
                    # Flat anomaly_score listesi okunur.
                    anomaly_scores = r.get("anomaly_score", [])
                    if isinstance(anomaly_scores, list):
                        for p in anomaly_scores:
                            if isinstance(p, dict):
                                v = safe_float(p.get("value"), 0.0)
                                if v > score:
                                    score = v
                    # Flat anomalies listesi okunur.
                    anomalies = r.get("anomalies", [])
                    if isinstance(anomalies, list):
                        for a in anomalies:
                            if isinstance(a, dict):
                                err = safe_float(a.get("error"), 0.0)
                                if err > score:
                                    score = err
            # Toplam anomali, maksimum score ve threshold döndürülür.
            return total_anoms, score, threshold

        # REQUEST PAYLOAD OKUMA
        # WebDev request içindeki postData alınır.
        raw = request.get("postData")
        # Raw postData loglanır.
        log.info("RAW POSTDATA = %s" % raw)
        # postData tipi loglanır.
        log.info("POSTDATA TYPE = %s" % type(raw))
        # postData string ise JSON decode yapılır.
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
        # Backend adresi loglanır.
        log.info("backend = %s" % payload.get("backend"))
        # Seçili tag path loglanır.
        log.info("tagPath = %s" % payload.get("tagPath"))
        # Eğitim süresi loglanır.
        log.info("trainMinutes = %s" % payload.get("trainMinutes"))
        # Inference süresi loglanır.
        log.info("inferMinutes = %s" % payload.get("inferMinutes"))
        # UI threshold değeri loglanır.
        log.info("threshold = %s" % payload.get("threshold"))
        # Zorunlu payload alanları kontrol edilir.
        for required_key in ["backend", "tagPath", "trainMinutes", "inferMinutes", "threshold"]:
            # Eksik zorunlu alan varsa 400 hata döndürülür.
            if required_key not in payload or payload.get(required_key) is None:
                return return_error_response(
                    "Missing required payload field: %s" % required_key,
                    400,
                    payload,
                    None,
                    None,
                    0
                )
        # Holdout minutes payload içinden alınır; yoksa inferMinutes kullanılır.
        holdoutMinutes = payload.get("holdoutMinutes", payload.get("inferMinutes"))
        # Step seconds payload içinden alınır; yoksa 5 saniye kullanılır.
        stepSeconds = payload.get("stepSeconds", 5)
        # Holdout değeri loglanır.
        log.info("holdoutMinutes = %s" % holdoutMinutes)
        # Step seconds değeri loglanır.
        log.info("stepSeconds = %s" % stepSeconds)
        # Batch minutes payload içinden alınır.
        batchMinutes = payload.get("batchMinutes", None)
        # Batch stride yoksa batchMinutes ile aynı alınır.
        batchStrideMinutes = payload.get("batchStrideMinutes", batchMinutes)
        # Batch minutes loglanır.
        log.info("batchMinutes = %s" % batchMinutes)
        # Batch stride minutes loglanır.
        log.info("batchStrideMinutes = %s" % batchStrideMinutes)
        # anomaly.py içindeki BACKEND global değişkeni UI'dan gelen backend ile güncellenir.
        anomaly.BACKEND = payload["backend"]
        # Backend adresinin set edildiği loglanır.
        log.info("BACKEND SET = %s" % anomaly.BACKEND)

        # TRAIN ÇAĞRISI
        # WebDev endpoint önce trainTags çağırır.
        train_resp = anomaly.trainTags([payload["tagPath"]], minutes=payload["trainMinutes"])
        # train response tipi loglanır.
        log.info("TYPE TRAIN = %s" % type(train_resp))
        # train response içeriği loglanır.
        log.info("TRAIN RESP = %s" % train_resp)
        # Train tarafı error döndürdüyse infer çağırmadan kullanıcıya hata döndürülür.
        if isinstance(train_resp, dict) and train_resp.get("error"):
            # Hata mesajı train response içinden okunur.
            msg = train_resp.get("detail") or train_resp.get("user_message") or "There is no enough data for training."
            # Train fail loglanır.
            log.error("TRAIN failed before infer: %s" % train_resp)
            # Hata tagleri güncellenir.
            safe_write(
                [
                    "[default]ml/Backend_Status",
                    "[default]ml/Backend_ErrorMessage",
                    "[default]ml/HttpStatusCode",
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
                    400,
                    False,
                    "NORMAL",
                    system.util.jsonEncode(payload),
                    system.util.jsonEncode({
                        "train": to_json_safe(train_resp),
                        "infer": None
                    }),
                    system.date.now(),
                    False
                ],
                "TRAIN EARLY ERROR TAG WRITE"
            )
            # WebDev JSON response döndürülür.
            return {
                "json": {
                    "ok": False,
                    "user_message": msg,
                    "error": {
                        "train": train_resp,
                        "infer": None
                    }
                }
            }

        # INFER ÇAĞRISI
        # Infer başlangıç zamanı milisaniye olarak alınır.
        infer_start_ms = system.date.now().getTime()
        # anomaly.py içindeki inferTags çağrılır.
        infer_resp = anomaly.inferTags(
            [payload["tagPath"]],
            minutes=payload["inferMinutes"],
            thr=payload["threshold"],
            holdout_minutes=holdoutMinutes,
            step_seconds=stepSeconds,
            batch_minutes=batchMinutes,
            batch_stride_minutes=batchStrideMinutes
        )
        # Infer latency hesaplanır.
        infer_latency_ms = int(system.date.now().getTime() - infer_start_ms)
        # infer response tipi loglanır.
        log.info("TYPE INFER = %s" % type(infer_resp))
        # infer response içeriği loglanır.
        log.info("INFER RESP = %s" % infer_resp)
        # Train response JSON-safe hale getirilir.
        train_safe = to_json_safe(train_resp)
        # Infer response JSON-safe hale getirilir.
        infer_safe = to_json_safe(infer_resp)

        # TRAIN / INFER HATA KONTROLÜ
        # Train veya infer response içinde error varsa hata akışına girilir.
        if (isinstance(train_resp, dict) and "error" in train_resp) or (isinstance(infer_resp, dict) and "error" in infer_resp):
            # Hata loglanır.
            log.error("TRAIN or INFER failed: %s %s" % (train_resp, infer_resp))
            # Default hata mesajı.
            msg = "Model process error"
            # Default HTTP status.
            http_status = 500
            # Infer tarafı backend_non_2xx döndürdüyse HTTP status ve mesaj infer_resp'ten alınır.
            if isinstance(infer_resp, dict) and infer_resp.get("error") == "backend_non_2xx":
                http_status = int(infer_resp.get("status", 500))
                if infer_resp.get("detail"):
                    msg = "Infer failed: %s" % infer_resp.get("detail")
                else:
                    msg = "Infer failed (HTTP %s)" % http_status
            # Train tarafı backend_non_2xx döndürdüyse HTTP status ve mesaj train_resp'ten alınır.
            elif isinstance(train_resp, dict) and train_resp.get("error") == "backend_non_2xx":
                http_status = int(train_resp.get("status", 500))
                if train_resp.get("detail"):
                    msg = "Train failed: %s" % train_resp.get("detail")
                else:
                    msg = "Train failed (HTTP %s)" % http_status
            # Infer network/timeout hatası varsa kullanıcı dostu mesaj set edilir.
            elif isinstance(infer_resp, dict) and infer_resp.get("error") in ["network_timeout", "backend_unreachable"]:
                msg = infer_resp.get("detail") or "FastAPI server could not be reached."
                http_status = 503
            # Train network/timeout hatası varsa kullanıcı dostu mesaj set edilir.
            elif isinstance(train_resp, dict) and train_resp.get("error") in ["network_timeout", "backend_unreachable"]:
                msg = train_resp.get("detail") or "FastAPI server could not be reached."
                http_status = 503
            # Diğer infer hataları için detail kullanılır.
            elif isinstance(infer_resp, dict) and infer_resp.get("detail"):
                msg = infer_resp.get("detail")
            # Diğer train hataları için detail kullanılır.
            elif isinstance(train_resp, dict) and train_resp.get("detail"):
                msg = train_resp.get("detail")
            # Standart hata response'u döndürülür.
            return return_error_response(msg, http_status, payload, train_safe, infer_safe, infer_latency_ms)

        # INFER ÖZET BİLGİSİ
        # Infer response'tan toplam anomali sayısı, son/en yüksek score ve threshold çekilir.
        total_anoms, last_score, last_threshold = extract_infer_summary(infer_resp)
        # Eğer response içinden threshold çıkmazsa UI’dan gelen threshold fallback olarak kullanılır.
        if last_threshold == 0.0:
            last_threshold = safe_float(payload.get("threshold"), 0.0)
        # Toplam anomali sayısı 0'dan büyükse son durum anomaly kabul edilir.
        actual_is_anomaly = total_anoms > 0
        # Son tahmin text'i oluşturulur.
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
        # ML tagleri güncellenir.
        safe_write(ml_paths, ml_values, "ML SUCCESS TAG WRITE")
        # SON TEST SONUCUNU GÜNCELLE
        # Test beklenen sonuç ve evidence bilgileri okunacak tag pathleri.
        result_read_paths = [
            "[default]results/TestCaseName",
            "[default]results/Expected_IsAnomaly",
            "[default]results/Expected_Status",
            "[default]evidence/LastSensorName",
            "[default]evidence/LastInputValue"
        ]
        # Test/evidence tagleri okunur.
        result_read_values = system.tag.readBlocking(result_read_paths)
        # Test case adı güvenli şekilde okunur.
        test_case_name = str(result_read_values[0].value) if result_read_values[0].quality.isGood() and result_read_values[0].value is not None else "NONE"
        # Beklenen anomaly boolean değeri okunur.
        expected_is_anomaly = bool(result_read_values[1].value) if result_read_values[1].quality.isGood() else False
        # Beklenen status okunur.
        expected_status = str(result_read_values[2].value) if result_read_values[2].quality.isGood() and result_read_values[2].value is not None else "NORMAL"
        # Son sensör adı okunur.
        last_sensor_name = str(result_read_values[3].value) if result_read_values[3].quality.isGood() and result_read_values[3].value is not None else "NONE"
        # Son input value okunur.
        last_input_value = safe_float(result_read_values[4].value, 0.0)
        # Test case tanımlıysa test sonucu hesaplanır.
        if test_case_name != "NONE":
            # Dropout handled senaryosu özel başarılı kabul edilir.
            if expected_status == "HANDLED_DROPOUT":
                test_passed = True
            # Diğer senaryolarda beklenen anomaly ile actual anomaly karşılaştırılır.
            else:
                test_passed = (expected_is_anomaly == actual_is_anomaly)
            # PASS/FAIL text'i oluşturulur.
            result_text = "PASS" if test_passed else "FAIL"
            # Test mesajı kullanıcıya/kanıta yazılmak üzere oluşturulur.
            test_message = "Backend infer completed. Sensor=%s, Input=%.4f, Expected=%s, Actual=%s, AnomalyCount=%s, Score=%.6f, Threshold=%.6f" % (
                last_sensor_name,
                last_input_value,
                expected_status,
                actual_status,
                total_anoms,
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
            # Test sonucu tagleri güncellenir.
            safe_write(result_write_paths, result_write_values, "RESULT TAG WRITE")
        # BAŞARILI RESPONSE
        # Train response queued True ise başarılı sonuç döndürülür.
        if isinstance(train_resp, dict) and train_resp.get("queued") == True:
            # Başarılı bitiş loglanır.
            log.info("WEBDEV TRAIN ENDED SUCCESSFULLY")
            # Perspective/UI tarafına train ve infer response'u döndürülür.
            return {
                "json": {
                    "ok": True,
                    "train": train_resp,
                    "infer": infer_resp
                }
            }
        # Eğer train_resp queued True değilse yine de güvenli başarılı response döndürülür.
        log.info("WEBDEV TRAIN ENDED WITH NON-QUEUED SUCCESS RESPONSE")
        return {
            "json": {
                "ok": True,
                "train": train_resp,
                "infer": infer_resp
            }
        }

    # GLOBAL CRASH HANDLER
    # Beklenmeyen herhangi bir hata olursa burası çalışır.
    except Exception as e:
        # Detaylı traceback string olarak alınır.
        err_msg = traceback.format_exc()
        # Logger tekrar alınır; try bloğu başında log oluşmadan hata olduysa garanti eder.
        log = system.util.getLogger("WEBDEV-TRAIN")
        # Crash detayları loglanır.
        log.error("CRASH: " + err_msg)
        # Hatalı bitiş loglanır.
        log.info("WEBDEV TRAIN ENDED WITH ERROR")
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