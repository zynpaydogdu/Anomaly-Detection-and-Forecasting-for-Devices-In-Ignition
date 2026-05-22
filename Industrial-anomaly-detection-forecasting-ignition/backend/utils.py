import numpy as np
import os
import csv
from datetime import datetime, timedelta

# Dosya yolu yardımcı fonksiyonları
def mp(tag): return f"models/{tag.replace('/', '_')}.pt"
# Ana model dosyası (PyTorch .pt) için path üretir
# Örn tag="line1/sensorA" -> "models/line1_sensorA.pt"

def mp_ae(tag): return f"models/{tag.replace('/', '_')}_ae.pt"
# Autoencoder (AE) modeli için path üretir
# Örn "models/line1_sensorA_ae.pt"

def sp(tag): return f"models/{tag.replace('/', '_')}_scaler.joblib"
# Scaler dosyası için path üretir
# Bu scaler, eğitimde kullanılan min-max değerlerini saklar.
# Örn "models/line1_sensorA_scaler.joblib"


# CSV kayıt fonksiyonu
# Amaç Eğitim/inceleme için gelen zaman serisini "datasets/" altına "timestamp,value" kolonları ile kaydetmek.
def save_csv(tag, timestamps, values):
   # Tag'den güvenli bir CSV yolu oluştu
   path = f"datasets/{tag.replace('/', '_')}.csv"

   
    # "w" dosyayı sıfırdan yazar (varsa üzerine yazar)
    # newline='' -> Windows’ta ekstra boş satır oluşmasını önler
    # encoding="utf-8" -> Türkçe karakter sorun çıkarmasın
   with open(path, "w", newline='', encoding="utf-8") as f:
       writer = csv.writer(f)
       # CSV başlık satırı
       writer.writerow(["timestamp", "value"])
       
        # timestamps ve values listelerini aynı anda dolaşır
        # zip: (t0,v0), (t1,v1) şeklinde eşleşme sağlar
       for t, v in zip(timestamps, values):
           writer.writerow([t, v])


# Sliding Window oluşturma
# Amaç Zaman serisini, modelin öğrenebileceği X (girdi) ve y (hedef)
# çiftlerine çevirmek.
# Örn
# arr = [10, 11, 12, 13, 14], look_back = 3
# X = [[10,11,12], [11,12,13]]
# y = [13, 14]
# Sonuç şekli
# X -> (num_samples, look_back, 1)
# y -> (num_samples, 1)
def make_windows(arr, look_back):
   X, y = [], []
   
    # look_back kadar geçmişe bakacağımız için
    # en son look_back eleman pencere oluşturamaz, o yüzden
   for i in range(len(arr) - look_back):
       # X: look_back uzunlukta geçmiş değerler
       X.append(arr[i: i + look_back])
       # y: pencerenin hemen sonraki gerçek değeri (tahmin hedefi)
       y.append(arr[i + look_back])
       
    # Model (özellikle RNN/LSTM gibi) genelde float32 ister
    # reshape(-1, look_back, 1) her pencereyi (look_back, 1) yapar
   X = np.array(X, dtype=np.float32).reshape(-1, look_back, 1)
   y = np.array(y, dtype=np.float32).reshape(-1, 1)
   return X, y

#MinMaxScaler
# Amaç Veriyi 0-1 aralığına sıkıştırmak (normalize etmek).
# formül
# scaled = (x - min) / (max - min)
# Not: max == min ise (tüm değerler aynıysa) bölme hatası olmaması için range_ = 1.0 yapılıyor.
class MinMaxScaler:
   def fit(self, arr):
       # Verinin minimum ve maksimum değerlerini bulur
       self.min_ = arr.min()
       self.max_ = arr.max()
       # Aralık sıfırsa (min==max) range'i 1 yapıp güvenli hale getirir
       self.range_ = self.max_ - self.min_ if self.max_ != self.min_ else 1.0
       return self
   def transform(self, arr):
       # Fit sırasında öğrenilen min/range ile normalize eder
       return (arr - self.min_) / self.range_
   def inverse_transform(self, arr):
       # Normalize edilmiş değeri tekrar orijinal ölçeğe geri çevirir
       return arr * self.range_ + self.min_
   

# Threshold hesaplama
# Amaç Residual (hata) değerlerinden anomali eşiği belirlemek.
# İki farklı yaklaşım kullanılıyor:
# 1) Z-score temelli: mean + 3*std  (normal dağılım varsayımı gibi)
# 2) IQR temelli: Q3 + 1.5*IQR     (aykırı değer yaklaşımı)
# En sonda daha "temkinli" olması için ikisinin küçüğünü döndürüyor.
def compute_threshold(residuals):
   # Ortalama ve standart sapma ile z-score eşiği
   mean, std = residuals.mean(), residuals.std()
   z_thr = mean + 3 * std
   # IQR ile eşik
   Q1, Q3 = np.percentile(residuals, 25), np.percentile(residuals, 75)
   iqr_thr = Q3 + 1.5 * (Q3 - Q1)
   # İki eşiğin daha küçük olanı genelde daha az agresif/temkinli seçim
   return float(min(z_thr, iqr_thr))


# Gelecek zaman damgaları üretme
# Amaç Son timestamp'ten sonra n_steps kadar yeni timestamp üretmek.
# last_ts farklı formatlarda gelebilir:
# - Unix timestamp (saniye veya milisaniye) string/number gibi
# - "2026-04-29T12:30:00" gibi ISO format
# - "2026-04-29 12:30:00" gibi
# - "2026-04-29" gibi sadece tarih
# step_seconds: Her adımda kaç saniye ileri gidileceği (varsayılan 1 sn)
def generate_future_timestamps(last_ts, n_steps, step_seconds=1):
   from datetime import datetime, timedelta
   dt = None # last_ts başarıyla okunursa datetime burada tutulacak
    #1) Önce "epoch time" olabilir diye sayıya çevirmeyi dene
   try:
       ts_float = float(last_ts)   
        # Eğer 13 haneliyse büyük ihtimalle milisaniye timestamp'tir.
        # Örn: 1710000000000 gibi
       if ts_float > 1e11: 
           ts_float /= 1000.0
       dt = datetime.fromtimestamp(ts_float)
       
    # last_ts sayıya çevrilemezse (örn "2026-04-29T12:30:00")
    # ValueError'a düşer ve string format denemelerine geçeriz.
   except ValueError:
       # Denenecek tarih formatları listesi
       formats = [
           "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
           "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d"
       ]
       # last_ts hangi formata uyuyorsa onu bulmaya çalı
       for fmt in formats:
           try:
               dt = datetime.strptime(str(last_ts), fmt)
               break
           except ValueError:
               continue
    
    # 2) Hâlâ okunamadıysa uyarı basıp şu an ile devam et (Bu, sistemin tamamen crash olmasını engeller.)
   if dt is None:
       print(f"DIKKAT: Tarih okunamadi: {last_ts}")
       dt = datetime.now()
    
    # 3) dt'den başlayarak n_steps kadar ileri zaman üret
    # Örn step_seconds=60 ise her seferinde +1 dakika ilerler
   return [(dt + timedelta(seconds=(i + 1) * step_seconds)).strftime("%Y-%m-%dT%H:%M:%S") for i in range(n_steps)]