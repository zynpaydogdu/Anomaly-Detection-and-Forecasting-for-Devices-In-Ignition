# JSON formatına çevirmek ve JSON string üretmek için kullanılır.
import json
# Java taraflı network/timeout hatalarını yakalayabilmek için JavaException alias'ı kullanılır.
from java.lang import Exception as JavaException
# Bu script modülüne ait loglar Gateway Logs altında IF-API etiketiyle görünür.
IF_LOG = system.util.getLogger("IF-API")
# FastAPI backend servis adresi.
# Bu kod Ignition Gateway üzerinde çalıştığı için localhost/127.0.0.1 Gateway makinesini ifade eder.
# FastAPI başka bir bilgisayarda çalışıyorsa buraya o bilgisayarın IP adresi yazılmalıdır.
BACKEND = "http://localhost:8000"
# HTTP client oluşturmak için ortak yardımcı fonksiyon.

def _client(timeout=600000):
   # Ignition HTTP client nesnesi oluşturulur.
   return system.net.httpClient(timeout=timeout)
   
# Tag listesindeki tekrarları temizlemek ve her değeri string yapmak için yardımcı fonksiyon.
def _unique_tags(tagList):
   # Gelen tüm tag değerleri string'e çevrilir.
   tagList = [str(t) for t in tagList]
   # dict.fromkeys sıralamayı koruyarak duplicate değerleri temizler.
   return list(dict.fromkeys(tagList))
   
# HTTP response gövdesini güvenli şekilde JSON'a çevirmek için yardımcı fonksiyon.
def _response_json_or_ok(resp, endpoint_name):
   # Önce Ignition response nesnesinden JSON parse etmeyi dener.
   try:
       # Response JSON ise direkt dict/list olarak döndürür.
       return resp.getJson()
   # JSON parse başarısız olursa fallback response döndürülür.
   except Exception as e:
       # Parse hatası loglanır.
       IF_LOG.error("%s getJson hatasi | %s" % (endpoint_name, e))
       # JSON parse edilemese bile HTTP başarılıysa minimum başarılı response döndürülür.
       return {"ok": True, "status": resp.getStatusCode(), "body": resp.getText()}
       
# 2xx olmayan HTTP cevaplarını standart hata formatına çevirmek için yardımcı fonksiyon.
def _backend_non_2xx(resp):
   # Response text alınır.
   txt = resp.getText()
   # Backend detail bilgisi başlangıçta None kabul edilir.
   detail = None
   # Backend body JSON ise detail/message alanı okunmaya çalışılır.
   try:
       # Ignition JSON decode ile response text dict'e çevrilir.
       j = system.util.jsonDecode(txt)
       # FastAPI genelde hata detayını detail alanında verir.
       detail = j.get("detail") or j.get("message")
   # Body JSON değilse detail None kalır.
   except:
       detail = None
   # Standart hata objesi döndürülür.
   return {
       "error": "backend_non_2xx",
       "status": resp.getStatusCode(),
       "detail": detail,
       "body": txt
   }
# FastAPI /health endpoint'ini kontrol eder.
def testHealth():
   # Health isteği atılacağı loglanır.
   IF_LOG.info("Health GET istegi atiliyor | backend=%s" % BACKEND)
   # Network hatalarını yakalamak için try kullanılır.
   try:
       # HTTP client oluşturulur.
       client = _client(timeout=10000)
       # FastAPI health endpoint'ine GET isteği atılır.
       resp = client.get(BACKEND + "/health")
       # HTTP status code loglanır.
       IF_LOG.info("Health response | status=%s body=%s" % (resp.getStatusCode(), resp.getText()))
       # Status 2xx ise bağlantı başarılı kabul edilir.
       return resp.getStatusCode() >= 200 and resp.getStatusCode() < 300
   # Java network/timeout hataları burada yakalanır.
   except JavaException as e:
       # Java bağlantı hatası loglanır.
       IF_LOG.error("Health JAVA NETWORK ERROR | %s" % e)
       # Health başarısız kabul edilir.
       return False
   # Diğer hatalar burada yakalanır.
   except Exception as e:
       # Genel bağlantı hatası loglanır.
       IF_LOG.error("Health baglanti hatasi | %s" % e)
       # Health başarısız kabul edilir.
       return False
       
# Belirtilen rootPath altındaki atomic tag'leri listeler.
def listTags(rootPath="[default]test"):
   # Verilen path altında tag browse işlemi yapılır.
   results = system.tag.browse(rootPath).getResults()
   # Bulunan tag path'leri burada tutulur.
   tags = []
   # Browse sonucu tek tek gezilir.
   for r in results:
       # Sadece gerçek değer tutan Atomic Tag'ler alınır.
       if "Atomic" in str(r["tagType"]):
           # Tag'in tam path'i listeye eklenir.
           tags.append(str(r["fullPath"]))
   # Örnek dönüş: ["[default]test/AnomalyTest1", "[default]test/AnomalyTest2"]
   return tags
   
# Tek bir tag için Tag History verisini çeker.
def historyForTag(tagPath, minutes=120, intervalSec=10):
   # Şu anki zaman aralığın bitiş noktasıdır.
   end = system.date.now()
   # Başlangıç zamanı, şu andan minutes kadar geriye gidilerek hesaplanır.
   start = system.date.addMinutes(end, -minutes)
   # İstenen örnek sayısı dakika ve örnekleme aralığına göre hesaplanır.
   return_size = int((minutes * 60) / intervalSec)
   # returnSize minimum 1 olmalıdır.
   if return_size < 1:
       # Çok küçük süre verilirse en az 1 satır istenir.
       return_size = 1
   # Tag history sorgusu denenir.
   try:
       # Kalitesi kötü verileri yok sayarak ve interpolasyon yapmadan history çekilir.
       ds = system.tag.queryTagHistory(
           paths=[tagPath],
           startDate=start,
           endDate=end,
           returnSize=return_size,
           aggregationMode="Average",
           ignoreBadQuality=True,
           noInterpolation=True
       )
   # Bazı Ignition sürümlerinde ignoreBadQuality/noInterpolation parametreleri sorun çıkarırsa fallback sorgu yapılır.
   except Exception as e:
       # İlk history sorgusunun fallback'e düştüğü loglanır.
       IF_LOG.error("HIST query fallback | tag=%s reason=%s" % (tagPath, e))
       # Daha sade parametrelerle tag history tekrar sorgulanır.
       ds = system.tag.queryTagHistory(
           paths=[tagPath],
           startDate=start,
           endDate=end,
           returnSize=return_size,
           aggregationMode="Average"
       )
   # Dönen dataset satır sayısı alınır.
   rowCount = ds.getRowCount()
   # History sorgu özeti loglanır.
   IF_LOG.info("HIST | tag=%s requested_minutes=%s intervalSec=%s requested_rows=%s actual_rows=%s" % (
       tagPath,
       minutes,
       intervalSec,
       return_size,
       rowCount
   ))
   # Timestamp listesi.
   ts = []
   # Value listesi.
   vs = []
   # Dataset satırları tek tek gezilir.
   for i in range(rowCount):
       # 0. kolon timestamp'tir.
       t = ds.getValueAt(i, 0)
       # 1. kolon tag value'dur.
       v = ds.getValueAt(i, 1)
       # Null değerler atlanır.
       if v is not None:
           # Değer float'a çevrilmeye çalışılır.
           try:
               # Sayısal değer float olarak alınır.
               fv = float(v)
               # Timestamp ISO benzeri string formatına çevrilir.
               ts.append(system.date.format(t, "yyyy-MM-dd'T'HH:mm:ss.SSS"))
               # Float değer value listesine eklenir.
               vs.append(fv)
           # Sayıya çevrilemeyen değerler atlanır.
           except Exception as e:
               # Parse edilemeyen satır debug için loglanır.
               IF_LOG.error("HIST value parse skip | tag=%s row=%s value=%s err=%s" % (tagPath, i, v, e))
   # Parse edilmiş veri sayısı loglanır.
   IF_LOG.info("HIST PARSED | tag=%s len(vs)=%s requested_rows=%s" % (
       tagPath,
       len(vs),
       return_size
   ))
   # Eğer veri geldiyse zaman aralığı loglanır.
   if len(ts) > 0:
       # İlk ve son timestamp loglanır.
       IF_LOG.info("HIST TIME RANGE | tag=%s first=%s last=%s len=%s" % (
           tagPath,
           ts[0],
           ts[-1],
           len(ts)
       ))
   # Hiç veri yoksa EMPTY loglanır.
   else:
       # Boş history sonucu loglanır.
       IF_LOG.info("HIST TIME RANGE | tag=%s EMPTY" % tagPath)
   # Timestamp ve value listeleri döndürülür.
   return ts, vs
   
# Bir veya daha fazla tag için history çekip FastAPI /train endpoint'ine gönderir.
def trainTags(tagList, minutes=120):
   # Train sürecinin başladığı loglanır.
   IF_LOG.info("TRAIN start | tags=%s minutes=%s BACKEND=%s" % (tagList, minutes, BACKEND))
   # Tag listesi boşsa işlem yapılamaz.
   if not tagList:
       # UI veya WebDev tarafına anlaşılır hata döndürülür.
       return {"error": "tagList is empty"}
   # Tag listesi string'e çevrilir ve duplicate değerler temizlenir.
   tags = _unique_tags(tagList)
   # FastAPI'ye gönderilecek series listesi.
   series = []
   # Her tag için history çekilir.
   for t in tags:
       # Eğitim için son minutes kadar history 5 saniye aralıkla çekilir.
       ts, vs = historyForTag(t, minutes=minutes, intervalSec=5)
       # Çekilen veri uzunluğu loglanır.
       IF_LOG.info("TRAIN HIST | tag=%s len(vs)=%s" % (t, len(vs)))
       # Minimum örnek sayısı belirlenir.
       min_samples = 30
       # Yeterli veri yoksa bu tag eğitime dahil edilmez.
       if len(vs) < min_samples:
           # Yetersiz veri loglanır.
           IF_LOG.error("TRAIN SKIP | tag=%s yeterli veri yok len(vs)=%s" % (t, len(vs)))
           # Sıradaki tag'e geçilir.
           continue
       # Veri minimumu hesaplanır.
       v_min = min(vs)
       # Veri maksimumu hesaplanır.
       v_max = max(vs)
       # Veri sabitse model eğitimi anlamsız olacağı için tag atlanır.
       if abs(v_max - v_min) < 0.000001:
           # Sabit/anlamsız veri loglanır.
           IF_LOG.error("TRAIN SKIP | tag=%s veri sabit veya anlamsız min=%s max=%s" % (t, v_min, v_max))
           # Sıradaki tag'e geçilir.
           continue
       # FastAPI'nin beklediği seri formatı oluşturulur.
       series.append({
           "tag": t,
           "timestamps": ts,
           "values": vs
       })
   # Hiçbir tag yeterli/verimli veri sağlayamadıysa backend'e istek atılmaz.
   if not series:
       # UI tarafına kullanıcı dostu hata döndürülür.
       return {
           "error": "not_enough_data",
           "detail": "There is no enough data for training."
       }
   # FastAPI /train endpoint'inin beklediği payload oluşturulur.
   payload = {"job_id": "ignition-auto-train", "series": series}
   # Payload JSON string'e çevrilir.
   jtxt = json.dumps(payload)
   # HTTP client oluşturulur.
   client = _client(timeout=600000)
   # Network/HTTP hatalarını yakalamak için try kullanılır.
   try:
       # Gönderilecek JSON boyutu loglanır.
       IF_LOG.info("POST /train sending | size=%s series_count=%s" % (len(jtxt), len(series)))
       # Debug için tüm JSON payload loglanır.
       # Çok büyük veriyle çalışırken bu satır Gateway Logs'u büyütebilir; gerekirse kapatılabilir.
       IF_LOG.info("POST /train payload = %s" % jtxt)
       # FastAPI /train endpoint'ine POST isteği gönderilir.
       resp = client.post(
           BACKEND + "/train",
           data=jtxt,
           headers={"Content-Type": "application/json"},
           timeout=600000
       )
       # HTTP durum kodu ve response body uzunluğu loglanır.
       IF_LOG.info("POST /train DONE | status=%s bodyLen=%s" % (resp.getStatusCode(), len(resp.getText())))
       # 2xx dışı response hata kabul edilir.
       if resp.getStatusCode() < 200 or resp.getStatusCode() >= 300:
           # Backend hata formatı standart hale getirilir.
           return _backend_non_2xx(resp)
       # 2xx ise response JSON olarak döndürülür.
       return _response_json_or_ok(resp, "POST /train")
   # Java network ve timeout hataları burada yakalanır.
   except JavaException as e:
       # Backend'e ulaşılamadıysa loglanır.
       IF_LOG.error("POST /train JAVA NETWORK ERROR | %s" % e)
       # UI tarafına kullanıcı dostu hata döndürülür.
       return {"error": "network_timeout", "detail": "FastAPI server could not be reached."}
   # Genel Python/Jython hataları burada yakalanır.
   except Exception as e:
       # Beklenmeyen hata loglanır.
       IF_LOG.error("POST /train EXCEPTION | %s" % e)
       # UI tarafına hata detayı döndürülür.
       return {"error": "backend_unreachable", "detail": str(e)}
       
# Eğitilmiş modellerle anomali/forecast için FastAPI /infer endpoint'ini çağırır.
def inferTags(tagList, minutes=30, thr=-0.5, holdout_minutes=None, step_seconds=None, batch_minutes=None, batch_stride_minutes=None, **kwargs):
   # Infer sürecinin başladığı loglanır.
   IF_LOG.info("INFER start | tags=%s minutes=%s thr=%s BACKEND=%s" % (tagList, minutes, thr, BACKEND))
   # Tag listesi boşsa işlem yapılamaz.
   if not tagList:
       # UI/WebDev tarafına hata döndürülür.
       return {"error": "tagList boş"}
   # camelCase parametre desteği: batch_minutes yoksa batchMinutes okunur.
   if batch_minutes is None:
       # WebDev/Perspective camelCase gönderirse burada yakalanır.
       batch_minutes = kwargs.get("batchMinutes")
   # camelCase parametre desteği: batch_stride_minutes yoksa batchStrideMinutes okunur.
   if batch_stride_minutes is None:
       # Verilmezse tumbling batch mantığı için stride=batch_minutes kabul edilir.
       batch_stride_minutes = kwargs.get("batchStrideMinutes", batch_minutes)
   # Tag listesi string'e çevrilir ve duplicate değerler temizlenir.
   tags = _unique_tags(tagList)
   # holdout_minutes verilmezse infer minutes ile aynı kabul edilir.
   if holdout_minutes is None:
       # Default holdout değeri minutes olur.
       holdout_minutes = minutes
   # step_seconds verilmezse default 5 saniye kullanılır.
   if step_seconds is None:
       # Sensör verin 5 saniyede bir geliyorsa doğru ayardır; 1 saniye ise 1 yapılabilir.
       step_seconds = 5
   # Eğer infer minutes holdout'tan küçükse yeterli pencere oluşmaz.
   if minutes < holdout_minutes:
       # minutes değeri holdout_minutes'a yükseltilir.
       IF_LOG.info("minutes (%s) < holdout_minutes (%s) -> minutes := holdout_minutes" % (minutes, holdout_minutes))
       # minutes güncellenir.
       minutes = holdout_minutes
   # FastAPI'ye gönderilecek seri listesi.
   series = []
   # Her tag için history çekilir.
   for t in tags:
       # Inference için son minutes kadar history, step_seconds aralığıyla çekilir.
       ts, vs = historyForTag(t, minutes=minutes, intervalSec=step_seconds)
       # Çekilen infer verisi uzunluğu loglanır.
       IF_LOG.info("INFER HIST | tag=%s len(vs)=%s" % (t, len(vs)))
       # Minimum örnek sayısı belirlenir.
       min_samples = 30
       # Yeterli veri yoksa bu tag atlanır.
       if len(vs) < min_samples:
           # Yetersiz veri loglanır.
           IF_LOG.error("INFER SKIP | tag=%s yeterli veri yok len(vs)=%s" % (t, len(vs)))
           # Sıradaki tag'e geçilir.
           continue
       # Veri minimumu hesaplanır.
       v_min = min(vs)
       # Veri maksimumu hesaplanır.
       v_max = max(vs)
       # Veri sabitse inference sonucu anlamlı olmayacağı için atlanır.
       if abs(v_max - v_min) < 0.000001:
           # Sabit/anlamsız veri loglanır.
           IF_LOG.error("INFER SKIP | tag=%s veri sabit veya anlamsız min=%s max=%s" % (t, v_min, v_max))
           # Sıradaki tag'e geçilir.
           continue
       # FastAPI'nin beklediği seri formatı oluşturulur.
       series.append({
           "tag": t,
           "timestamps": ts,
           "values": vs
       })
   # Hiç geçerli seri oluşmadıysa backend'e istek atılmaz.
   if not series:
       # UI tarafına kullanıcı dostu hata döndürülür.
       return {"error": "no_data", "detail": "Infer için yeterli veri bulunamadı."}
   # FastAPI /infer endpoint'ine gönderilecek meta parametreleri.
   meta = {
       "score_threshold": float(thr),
       "holdout_minutes": int(holdout_minutes),
       "step_seconds": int(step_seconds),
       "anomaly_confirm_points": 1,
       "recover_points": 6,
       "range_margin": 0.03
   }
   # batch_minutes verildiyse meta içine eklenir.
   if batch_minutes is not None:
       # Değer int'e çevrilmeye çalışılır.
       try:
           # Batch süresi dakika cinsinden meta'ya eklenir.
           meta["batch_minutes"] = int(batch_minutes)
       # Çevrilemezse sessizce atlanır.
       except Exception as e:
           # Geçersiz batch_minutes loglanır.
           IF_LOG.error("Invalid batch_minutes | value=%s err=%s" % (batch_minutes, e))
   # batch_stride_minutes verildiyse meta içine eklenir.
   if batch_stride_minutes is not None:
       # Değer int'e çevrilmeye çalışılır.
       try:
           # Batch stride süresi dakika cinsinden meta'ya eklenir.
           meta["batch_stride_minutes"] = int(batch_stride_minutes)
       # Çevrilemezse sessizce atlanır.
       except Exception as e:
           # Geçersiz batch_stride_minutes loglanır.
           IF_LOG.error("Invalid batch_stride_minutes | value=%s err=%s" % (batch_stride_minutes, e))
   # FastAPI /infer endpoint'inin beklediği payload oluşturulur.
   payload = {
       "request_id": "ignition-infer",
       "series": series,
       "meta": meta
   }
   # HTTP client oluşturulur.
   client = _client(timeout=600000)
   # Payload JSON string'e çevrilir.
   jtxt = json.dumps(payload)
   # Network/HTTP hatalarını yakalamak için try kullanılır.
   try:
       # Gönderilecek JSON boyutu ve meta bilgisi loglanır.
       IF_LOG.info("POST /infer sending | size=%s | series_count=%s | meta=%s" % (len(jtxt), len(series), meta))
       # FastAPI /infer endpoint'ine POST isteği gönderilir.
       resp = client.post(
           BACKEND + "/infer",
           data=jtxt,
           headers={"Content-Type": "application/json"},
           timeout=600000
       )
       # HTTP status code ve body uzunluğu loglanır.
       IF_LOG.info("POST /infer DONE | status=%s bodyLen=%s" % (resp.getStatusCode(), len(resp.getText())))
       # 2xx dışı response hata kabul edilir.
       if resp.getStatusCode() < 200 or resp.getStatusCode() >= 300:
           # Backend hata response'u standart hale getirilir.
           return _backend_non_2xx(resp)
       # 2xx response JSON olarak döndürülür.
       return _response_json_or_ok(resp, "POST /infer")
   # Java network ve timeout hataları burada yakalanır.
   except JavaException as e:
       # Backend'e ulaşılamadıysa loglanır.
       IF_LOG.error("POST /infer JAVA NETWORK ERROR | %s" % e)
       # UI tarafına kullanıcı dostu hata döndürülür.
       return {"error": "network_timeout", "detail": "FastAPI server could not be reached."}
   # Genel Python/Jython hataları burada yakalanır.
   except Exception as e:
       # Beklenmeyen infer hatası loglanır.
       IF_LOG.error("POST /infer EXCEPTION | %s" % e)
       # UI tarafına hata detayı döndürülür.
       return {"error": "backend_unreachable", "detail": str(e)}