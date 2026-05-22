from pydantic import BaseModel
from typing import List, Optional

class Series(BaseModel):
    # Bu sınıf TEK bir zaman serisini temsil eder.
    # Yani: "hangi veri?" + "hangi zamanlarda ölçüldü?" + "değerleri ne?"
    tag: str
    # Zaman serisinin adı/etiketi.
    # Örnek "sicaklik", "basinc", "sensor_12", "motor_rpm"
    timestamps: List[str]
    # Ölçüm yapılan zamanların listesi.
    # Her eleman bir timestamp olmalı.
    values: List[float]
    # timestamps listesindeki her zaman için ölçülen değerler.
    # Örn timestamps[0] zamanında ölçülen değer values[0] olur.Yani index index eşleşir

class TrainPayload(BaseModel):
    # Model trainin) için API'ye gönderilecek ana paket.
    job_id: str
    # Eğitim işini takip etmek için benzersiz kimlik.
    # Örnek: "job_001", "train_20260429_01"
    series: List[Series]
    # Eğitime gönderilen zaman serileri.
    # Birden fazla seri aynı anda gönderilebilir.
    # Örn aynı makinadan gelen 3 sensör verisi gibi.
    meta: Optional[dict] = None
    # Opsiyonel: Ek bilgi alanı.
    # Buraya eğitimle ilgili yardımcı bilgiler koyarsın.
    # Örnek
    # {
    #   "model": "lstm",
    #   "horizon": 24,
    #   "site": "factory-A",
    #   "note": "veri normalize edildi"
    # }

class InferPayload(BaseModel):
    # inference için API'ye gönderilecek paket.
    request_id: str
    # Bu tahmin isteğini takip etmek için benzersiz istek kimliği.
    # Örnek: "req_123", "infer_20260429_99"
    series: List[Series]
    # Tahmin yapılacak zaman serileri.
    meta: Optional[dict] = None
    # Opsiyonel: Tahmin sırasında kullanılacak ek ayarlar.
    # Örnek
    # {
    #   "return_confidence": True,
    #   "threshold": 0.8,
    #   "horizon": 12
    # }
