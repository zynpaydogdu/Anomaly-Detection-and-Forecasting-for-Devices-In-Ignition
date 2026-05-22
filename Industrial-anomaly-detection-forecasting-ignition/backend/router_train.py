# FastAPI router yapısını ve HTTP hatası fırlatmak için HTTPException sınıfını içe aktarır.
from fastapi import APIRouter, HTTPException
# NumPy; sensör verilerini array'e çevirmek, min/max/std hesaplamak ve veri kontrolü yapmak için kullanılır.
import numpy as np
# PyTorch; model eğitimi, tensor işlemleri ve model kaydetme/yükleme için kullanılır.
import torch
# PyTorch neural network modülü; loss fonksiyonları ve gradient clipping için kullanılır.
import torch.nn as nn
# DataLoader ve TensorDataset; eğitim verisini batch'lere bölmek için kullanılır.
from torch.utils.data import DataLoader, TensorDataset
# joblib; scaler nesnesini dosyaya kaydetmek ve daha sonra yüklemek için kullanılır.
import joblib
# os; model, autoencoder ve scaler dosyalarının var olup olmadığını kontrol etmek için kullanılır.
import os
# Train endpoint'ine gelen request body şeması.
from schemas import TrainPayload
# Forecast LSTM modeli, Autoencoder modeli, hyperparameter search fonksiyonu ve cihaz bilgisi içe aktarılır.
from models import LSTMModel, Autoencoder, hyperopt, DEVICE
# Yardımcı fonksiyonlar: model path, scaler path, csv kaydetme, window oluşturma ve MinMaxScaler.
from utils import mp, mp_ae, sp, save_csv, make_windows, MinMaxScaler
# Bu dosyada endpoint tanımlamak için FastAPI router oluşturulur.
router = APIRouter()

# GPU / CPU YARDIMCI FONKSİYONU
# Gelen veri tensor ise aktif cihaza taşır; tensor değilse olduğu gibi döndürür.
def to_device(x):
    # Eğer x PyTorch Tensor ise GPU varsa GPU'ya, yoksa CPU'ya taşınır.
    return x.to(DEVICE) if isinstance(x, torch.Tensor) else x

# MANUEL TRAINING LOOP
# Bu fonksiyon bir epoch boyunca modeli batch batch eğitir.
def train_loop(model, loader, criterion, optimizer):
    # Model training moduna alınır; dropout gibi katmanlar aktif olur.
    model.train()
    # Epoch boyunca toplam loss değerini tutar.
    total = 0
    # DataLoader içinden batch batch X ve y alınır.
    for Xb, yb in loader:
        # Input batch aktif cihaza taşınır.
        Xb = to_device(Xb)
        # Target batch aktif cihaza taşınır.
        yb = to_device(yb)
        # Önceki batch'ten kalan gradient değerleri sıfırlanır.
        optimizer.zero_grad()
        # Model output üretir ve gerçek değerle karşılaştırılarak loss hesaplanır.
        loss = criterion(model(Xb), yb)
        # Loss'a göre gradient hesaplanır.
        loss.backward()
        # Gradient clipping uygulanır; ani büyük gradient patlamalarını engeller.
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # Optimizer model parametrelerini günceller.
        optimizer.step()
        # Batch loss toplam loss'a eklenir.
        total += loss.item()
    # Epoch toplam loss değeri döndürülür.
    return total


# TRAIN ENDPOINT
# Ignition/WebDev tarafından gönderilen sensör verileriyle model eğitir.
@router.post("/train", status_code=202)
def train(p: TrainPayload):
    # Request içindeki opsiyonel meta alanı alınır; yoksa boş dict kullanılır.
    meta = p.meta or {}
    # Eğitim epoch sayısı meta içinden alınır; verilmezse 20 kullanılır.
    epochs = int(meta.get("epochs", 20))
    # Tuning açık mı kapalı mı belirlenir; default True.
    tuning = bool(meta.get("tuning", True))
    # force_retrain=False ise mevcut model varsa yeniden eğitim yapılmaz.
    # Bu, fault/drift/noise sırasında modelin anomalili veriyi normal davranış gibi öğrenmesini engeller.
    force_retrain = bool(meta.get("force_retrain", False))
    # Train request başlangıç bilgileri terminale yazdırılır.
    print("\n================ TRAIN REQUEST ================")
    # Eğitim işinin id bilgisi loglanır.
    print("job_id:", p.job_id)
    # Meta parametreleri loglanır.
    print("meta:", meta)
    # Epoch bilgisi loglanır.
    print("epochs:", epochs)
    # Tuning durumu loglanır.
    print("tuning:", tuning)
    # force_retrain durumu loglanır.
    print("force_retrain:", force_retrain)
    # Kaç adet seri/tag geldiği loglanır.
    print("series_count:", len(p.series))
    # Request başlangıç log bloğu kapatılır.
    print("================================================\n")
    # Her seri/tag için eğitim sonucunu tutacak liste.
    results = []
    # Request içinde birden fazla seri olabilir; her seri ayrı model olarak işlenir.
    for s in p.series:
        # O an işlenen serinin başlangıç logu.
        print("\n---------------- TRAIN SERIES ----------------")
        # Tag adı loglanır.
        print("tag:", s.tag)
        # Timestamp uzunluğu loglanır.
        print("timestamps_len:", len(s.timestamps))
        # Value uzunluğu loglanır.
        print("values_len:", len(s.values))
        # Gelen verinin debug istatistiklerini hesaplamayı dener.
        try:
            # Gelen values listesi float32 NumPy array'e çevrilir.
            arr_dbg = np.array(s.values, dtype=np.float32)
            # Veri varsa minimum değer loglanır.
            print("incoming_min:", float(arr_dbg.min()) if len(arr_dbg) else None)
            # Veri varsa maksimum değer loglanır.
            print("incoming_max:", float(arr_dbg.max()) if len(arr_dbg) else None)
            # Veri varsa ortalama değer loglanır.
            print("incoming_mean:", float(arr_dbg.mean()) if len(arr_dbg) else None)
            # Veri varsa standart sapma loglanır.
            print("incoming_std:", float(arr_dbg.std()) if len(arr_dbg) else None)
        # Debug sırasında hata olursa yakalanır.
        except Exception as e:
            # Debug hatası loglanır.
            print("incoming array debug failed:", str(e))
            # Model path debug için yazdırılır.
            print("model_path:", f_path)
            # Autoencoder path debug için yazdırılır.
            print("ae_path:", ae_path)
            # Scaler path debug için yazdırılır.
            print("scaler_path:", sc_path)
            # Forecast model dosyası var mı kontrol edilir.
            print("model_exists:", os.path.exists(f_path))
            # Autoencoder model dosyası var mı kontrol edilir.
            print("ae_exists:", os.path.exists(ae_path))
            # Scaler dosyası var mı kontrol edilir.
            print("scaler_exists:", os.path.exists(sc_path))
            # Debug log bloğu kapatılır.
            print("----------------------------------------------\n")
        
        # CACHE MEKANİZMASI
        # Forecast model dosya yolu oluşturulur.
        f_path = mp(s.tag)
        # Autoencoder model dosya yolu oluşturulur.
        ae_path = mp_ae(s.tag)
        # Scaler dosya yolu oluşturulur.
        sc_path = sp(s.tag)
        # Eğer forecast model, autoencoder model ve scaler zaten varsa ve force_retrain False ise eğitim atlanır.
        if os.path.exists(f_path) and os.path.exists(ae_path) and os.path.exists(sc_path) and not force_retrain:
            # Kayıtlı forecast model checkpoint'i yüklenir.
            checkpoint_f = torch.load(f_path, map_location="cpu")
            # Kayıtlı autoencoder model checkpoint'i yüklenir.
            checkpoint_ae = torch.load(ae_path, map_location="cpu")
            # Kayıtlı scaler yüklenir.
            scaler = joblib.load(sc_path)
            # Gelen yeni veri NumPy array'e çevrilir.
            arr = np.array(s.values, dtype=np.float32)
            # Cache ile ilgili uyarı başlangıçta yoktur.
            cache_warning = None
            # Eğer veri boş değilse scaler uyumluluk kontrolü yapılır.
            if len(arr) > 0:
                # Gelen verinin minimum ve maksimum değerleri hesaplanır.
                arr_min, arr_max = float(arr.min()), float(arr.max())
                # Scaler minimum değerini okumayı dener.
                try:
                    # Custom scaler veya sklearn scaler için min değeri güvenli şekilde okunur.
                    sc_min = float(np.array(getattr(scaler, "min_", scaler.data_min_)).reshape(-1)[0])
                # Okuma başarısız olursa fallback olarak min_ alanı denenir.
                except Exception:
                    # min_ yoksa 0.0 default kullanılır.
                    sc_min = float(getattr(scaler, "min_", 0.0))
                # Scaler maksimum değerini okumayı dener.
                try:
                    # Custom scaler veya sklearn scaler için max değeri güvenli şekilde okunur.
                    sc_max = float(np.array(getattr(scaler, "max_", scaler.data_max_)).reshape(-1)[0])
                # Okuma başarısız olursa fallback olarak max_ alanı denenir.
                except Exception:
                    # max_ yoksa 1.0 default kullanılır.
                    sc_max = float(getattr(scaler, "max_", 1.0))
                # Eğitim scaler aralığının %10'u kadar tolerans verilir.
                tol = 0.10 * abs(sc_max - sc_min)
                # Gelen veri eğitim scaler aralığının dışına çıktıysa warning oluşturulur.
                if arr_min < sc_min - tol or arr_max > sc_max + tol:
                    # Bu uyarı fault/anomaly ihtimalini belirtir; retrain yine de yapılmaz.
                    cache_warning = (
                        f"Incoming data is outside trained scaler range. "
                        f"This may indicate anomaly/fault, so retraining was skipped. "
                        f"ARR_MIN/MAX={arr_min:.4f}/{arr_max:.4f}, "
                        f"SCALER_MIN/MAX={sc_min:.4f}/{sc_max:.4f}"
                    )
                    # Terminale cache scaler mismatch uyarısı yazdırılır.
                    print(f"[WARN] {s.tag} cache scaler mismatch, but retrain skipped for anomaly safety.")
            # Cache kullanıldığını gösteren debug logu.
            print("\n TRAIN CACHE USED")
            # Cache kullanılan tag loglanır.
            print("tag:", s.tag)
            # force_retrain durumu loglanır.
            print("force_retrain:", force_retrain)
            # Varsa cache warning loglanır.
            print("cache_warning:", cache_warning)
            # Forecast model parametreleri loglanır.
            print("checkpoint_f_params:", checkpoint_f.get("params", {}))
            # Autoencoder model parametreleri loglanır.
            print("checkpoint_ae_params:", checkpoint_ae.get("params", {}))
            # Modelin eğitildiği verinin minimum değeri loglanır.
            print("checkpoint_train_min:", checkpoint_f.get("train_min", None))
            # Modelin eğitildiği verinin maksimum değeri loglanır.
            print("checkpoint_train_max:", checkpoint_f.get("train_max", None))
            # Modelin eğitildiği verinin ortalaması loglanır.
            print("checkpoint_train_mean:", checkpoint_f.get("train_mean", None))
            # Modelin eğitildiği verinin standart sapması loglanır.
            print("checkpoint_train_std:", checkpoint_f.get("train_std", None))
            # Autoencoder checkpoint içindeki threshold varsa loglanır.
            print("checkpoint_ae_threshold:", checkpoint_ae.get("threshold", None))
            # Threshold kaynağı varsa loglanır.
            print("checkpoint_threshold_source:", checkpoint_ae.get("threshold_source", None))
            # Bu tag için retrain yapılmayacağı loglanır.
            print("TRAIN WILL SKIP RETRAIN\n")
            # Response sonuç listesine cache kullanıldığı bilgisi eklenir.
            results.append({
                "tag": s.tag,
                "train_samples": "SKIPPED (cached normal model used)",
                "tuning": "SKIPPED",
                "force_retrain": force_retrain,
                "cache_warning": cache_warning,
                "lstm_params": checkpoint_f.get("params", {}),
                "ae_params": checkpoint_ae.get("params", {}),
            })
            # Bu tag için eğitim atlanır ve bir sonraki seriye geçilir.
            continue
        
        # YENİ MODEL EĞİTİMİNE HAZIRLIK
        # Gelen raw veri datasets klasörüne CSV olarak kaydedilir.
        save_csv(s.tag, s.timestamps, s.values)
        # Gelen values listesi float32 NumPy array'e çevrilir.
        arr = np.array(s.values, dtype=np.float32)
        # Veri uzunluğu hesaplanır.
        n = len(arr)
        
        # VERİ KALİTESİ KONTROLLERİ
        # Timestamp ve value uzunlukları eşleşmiyorsa veri bozuk kabul edilir.
        if len(s.timestamps) != len(s.values):
            # Kullanıcıya anlaşılır 400 hatası döndürülür.
            raise HTTPException(
                400,
                f"'{s.tag}' için timestamp/value uzunlukları eşleşmiyor. "
                f"timestamps={len(s.timestamps)}, values={len(s.values)}"
            )
        # Eğitim için minimum 30 örnek şartı kontrol edilir.
        if n < 30:
            # Yetersiz veri varsa model eğitimi durdurulur.
            raise HTTPException(
                400,
                f"'{s.tag}' için model eğitimi yapılamaz. En az 30 geçerli örnek gerekir, gelen örnek sayısı: {n}"
            )
        # NaN veya sonsuz değer kontrolü yapılır.
        if np.isnan(arr).any() or np.isinf(arr).any():
            # NaN/inf varsa model eğitimi iptal edilir.
            raise HTTPException(
                400,
                f"'{s.tag}' içinde NaN veya sonsuz değer var. Model eğitimi iptal edildi."
            )
        # Veri neredeyse sabitse model eğitimi anlamsız olur.
        if float(np.max(arr) - np.min(arr)) < 1e-6:
            # Flat/null veri için eğitim durdurulur.
            raise HTTPException(
                400,
                f"'{s.tag}' verisi sabit veya anlamsız görünüyor. Null/flat veri ile model eğitilemez."
            )
        
        # SCALING / NORMALIZATION
        # MinMaxScaler raw verinin min/max değerlerini öğrenir.
        scaler = MinMaxScaler().fit(arr)
        # Veri 0-1 aralığına normalize edilir.
        arr_scaled = scaler.transform(arr).astype(np.float32)
        # Yeni eğitim başlayacağı terminale yazdırılır.
        print("\n NEW TRAIN WILL RUN")
        # Eğitilecek tag loglanır.
        print("tag:", s.tag)
        # Veri uzunluğu loglanır.
        print("n:", n)
        # Raw minimum değer loglanır.
        print("raw_min:", float(arr.min()))
        # Raw maksimum değer loglanır.
        print("raw_max:", float(arr.max()))
        # Raw ortalama değer loglanır.
        print("raw_mean:", float(arr.mean()))
        # Raw standart sapma loglanır.
        print("raw_std:", float(arr.std()))
        # Scaler minimum değeri loglanır.
        print("scaler.min_:", float(scaler.min_))
        # Scaler maksimum değeri loglanır.
        print("scaler.max_:", float(scaler.max_))
        # Normalize verinin minimum değeri loglanır.
        print("scaled_min:", float(arr_scaled.min()))
        # Normalize verinin maksimum değeri loglanır.
        print("scaled_max:", float(arr_scaled.max()))
        # Eğitim başlangıcı loglanır.
        print(" TRAINING STARTED\n")
        
        # HYPERPARAMETER SEARCH İLE EĞİTİM
        # tuning True ise model parametreleri otomatik denenir.
        if tuning:
            # LSTM forecast modeli için en iyi model ve parametreler aranır.
            best_f_model, best_f_params, _ = hyperopt(arr_scaled, n, "lstm", epochs, DEVICE)
            # Autoencoder modeli için en iyi model ve parametreler aranır.
            best_ae_model, best_ae_params, _ = hyperopt(arr_scaled, n, "autoencoder", epochs, DEVICE)
            # Eğer uygun model bulunamazsa hata döndürülür.
            if best_f_model is None or best_ae_model is None:
                raise HTTPException(400, f"'{s.tag}' için kombinasyon bulunamadı")
            # Forecast modelinin en iyi look_back değeri alınır.
            look_back = best_f_params["look_back"]
            # Autoencoder modelinin en iyi look_back değeri alınır.
            look_back_ae = best_ae_params["look_back"]
        
        # MANUEL PARAMETRELERLE EĞİTİM
        # tuning False ise meta içinden gelen manuel parametrelerle eğitim yapılır.
        else:
            # LSTM'in kaç geçmiş noktaya bakacağı alınır.
            look_back = int(meta.get("look_back", 10))
            # LSTM hidden size değeri alınır.
            hidden_size = int(meta.get("hidden_size", 64))
            # Learning rate alınır.
            lr = float(meta.get("lr", 0.001))
            # Batch size alınır.
            batch_size = int(meta.get("batch_size", 16))
            # Autoencoder için aynı look_back kullanılır.
            look_back_ae = look_back
            # Veri look_back için yeterli değilse eğitim durdurulur.
            if n <= look_back + 1:
                raise HTTPException(400, f"'{s.tag}' için en az {look_back + 2} veri gerekir")
            # Normalize veriden forecast için X-y sliding window çiftleri oluşturulur.
            X_f, y_f = make_windows(arr_scaled, look_back)
            # X ve y PyTorch tensor formatına çevrilir.
            X_f_t, y_f_t = torch.tensor(X_f, dtype=torch.float32), torch.tensor(y_f, dtype=torch.float32)
            # Forecast modeli için DataLoader oluşturulur.
            loader_f = DataLoader(TensorDataset(X_f_t, y_f_t), batch_size=batch_size, shuffle=True)
            # Autoencoder input'u forecast X pencereleriyle aynı hizalanır.
            X_ae = X_f.copy()
            # Autoencoder input'u tensor formatına çevrilir.
            X_ae_t = torch.tensor(X_ae, dtype=torch.float32)
            # Autoencoder için input ve target aynı olacak şekilde DataLoader oluşturulur.
            loader_ae = DataLoader(TensorDataset(X_ae_t, X_ae_t), batch_size=batch_size, shuffle=True)
            # Forecast LSTM modeli oluşturulur ve aktif cihaza taşınır.
            best_f_model = LSTMModel(hidden_size=hidden_size).to(DEVICE)
            # Autoencoder modeli oluşturulur ve aktif cihaza taşınır.
            best_ae_model = Autoencoder(look_back).to(DEVICE)
            # Loss fonksiyonu olarak Mean Squared Error kullanılır.
            criterion = nn.MSELoss()
            # Forecast modeli için Adam optimizer oluşturulur.
            opt_f = torch.optim.Adam(best_f_model.parameters(), lr=lr)
            # Autoencoder modeli için Adam optimizer oluşturulur.
            opt_ae = torch.optim.Adam(best_ae_model.parameters(), lr=lr)
            # Belirlenen epoch sayısı kadar iki model de eğitilir.
            for _ in range(epochs):
                # Forecast LSTM bir epoch eğitilir.
                train_loop(best_f_model, loader_f, criterion, opt_f)
                # Autoencoder bir epoch eğitilir.
                train_loop(best_ae_model, loader_ae, criterion, opt_ae)
            # Manuel eğitimde forecast model parametreleri kaydedilecek dict'e yazılır.
            best_f_params = {
                "look_back": look_back,
                "hidden_size": hidden_size,
                "model_type": "lstm"
            }
            # Manuel eğitimde autoencoder model parametreleri kaydedilecek dict'e yazılır.
            best_ae_params = {
                "look_back": look_back_ae,
                "model_type": "autoencoder"
            }
        
        # MODEL, SCALER VE REFERANS BİLGİLERİNİ KAYDETME
        # Eğitim verisinin son normal penceresi reference_window olarak saklanır.
        # Infer sırasında fault/drift varsa prediction'ın doğrudan fault verisini takip etmesini engellemek için kullanılabilir.
        reference_window = arr_scaled[-look_back:].astype(np.float32).tolist()
        # Forecast LSTM modeli, parametreleri ve eğitim istatistikleri kaydedilir.
        torch.save({
            "model_state": best_f_model.cpu().state_dict(),
            "params": best_f_params,
            "reference_window": reference_window,
            "train_min": float(arr.min()),
            "train_max": float(arr.max()),
            "train_mean": float(arr.mean()),
            "train_std": float(arr.std()),
            "train_samples": int(n),
        }, mp(s.tag))
        # Autoencoder modeli ve parametreleri kaydedilir.
        torch.save({
            "model_state": best_ae_model.cpu().state_dict(),
            "params": best_ae_params
        }, mp_ae(s.tag))
        # Eğitimde kullanılan scaler kaydedilir.
        joblib.dump(scaler, sp(s.tag))
        # Bu tag için eğitim sonucu response listesine eklenir.
        results.append({
            "tag": s.tag,
            "train_samples": n,
            "tuning": tuning,
            "force_retrain": force_retrain,
            "lstm_params": best_f_params,
            "ae_params": best_ae_params,
        })
    # Tüm tag eğitim sonuçları response olarak döndürülür.
    return {
        "job_id": p.job_id,
        "queued": True,
        "results": results
    }