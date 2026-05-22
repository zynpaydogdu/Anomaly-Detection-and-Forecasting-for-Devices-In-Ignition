# FastAPI router yapısını ve HTTP hata fırlatma sınıfını içe aktarır.
from fastapi import APIRouter, HTTPException
# NumPy; array işlemleri, min/max/std hesapları ve pencereleme için kullanılır.
import numpy as np
# PyTorch; model yükleme, tensor oluşturma ve inference işlemleri için kullanılır.
import torch
# joblib; eğitimde kaydedilen scaler dosyasını yüklemek için kullanılır.
import joblib
# os; model/scaler dosyalarının varlığını kontrol etmek için kullanılır.
import os
# datetime; timestamp parse etmek ve milisaniye timestamp üretmek için kullanılır.
from datetime import datetime
# /infer endpoint’ine gelen request gövdesinin Pydantic şeması.
from schemas import InferPayload
# Eğitilmiş LSTM tahmin modeli ve LSTM Autoencoder modeli.
from models import LSTMModel, Autoencoder
# Yardımcı fonksiyonlar:
# mp      -> Forecast model path
# mp_ae   -> Autoencoder model path
# sp      -> Scaler path
# compute_threshold -> error değerlerinden threshold hesaplar
# generate_future_timestamps -> forecast timestamp üretir
from utils import mp, mp_ae, sp, compute_threshold, generate_future_timestamps
# Bu dosyada endpoint tanımlamak için FastAPI router oluşturulur.
router = APIRouter()

# TIMESTAMP PARSE FONKSİYONU
# Farklı formatlarda gelebilen timestamp değerlerini datetime objesine çevirir.
def _parse_dt(ts):
    try:
        # Timestamp sayısal gelmiş olabilir.
        # Örnek: 1710000000 veya 1710000000000
        x = float(ts)
        # Eğer sayı çok büyükse büyük ihtimalle milisaniye timestamp'tir.
        # Saniyeye çevirmek için 1000'e bölünür.
        if x > 1e11:
            x /= 1000.0
        # Unix timestamp datetime objesine çevrilir.
        return datetime.fromtimestamp(x)
    except Exception:
        # Sayıya çevrilemezse string format denemelerine geçilir.
        pass
    # Ignition veya backend’den gelebilecek farklı tarih formatları.
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d"
    ):
        try:
            # Timestamp verilen formata uyarsa datetime'a çevrilir.
            return datetime.strptime(str(ts), fmt)
        except Exception:
            # Format uymuyorsa bir sonraki format denenir.
            continue
    # Hiçbir format okunamazsa sistemin crash olmaması için şu anki zaman döndürülür.
    return datetime.now()

# STEP SECOND HESAPLAMA
# Timestamp listesinden verinin kaç saniyede bir geldiğini tahmin eder.
def _infer_step_seconds(ts_list):
    # En az 3 timestamp yoksa anlamlı aralık hesaplanamaz.
    if len(ts_list) < 3:
        return 1
    # Ardışık timestamp farkları burada tutulur.
    diffs = []
    # Tüm ardışık timestamp çiftleri gezilir.
    for i in range(1, len(ts_list)):
        # İki timestamp arasındaki saniye farkı hesaplanır.
        d = (_parse_dt(ts_list[i]) - _parse_dt(ts_list[i - 1])).total_seconds()
        # Sadece pozitif farklar alınır.
        if d > 0:
            diffs.append(d)
    # Hiç geçerli fark bulunamadıysa default 1 saniye kullanılır.
    if not diffs:
        return 1
    # Ortalama fark yuvarlanır ve en az 1 olacak şekilde döndürülür.
    return max(1, int(np.round(np.mean(diffs))))

# SCALER MIN/MAX OKUMA
# Custom scaler veya sklearn MinMaxScaler içinden min/max değerlerini güvenli okur.
def _get_scaler_minmax(scaler):
    # Başlangıçta min/max bilinmiyor kabul edilir.
    s_min = None
    s_max = None
    # Projede kullanılan custom MinMaxScaler min_ ve max_ alanlarına sahip.
    if hasattr(scaler, "min_") and hasattr(scaler, "max_"):
        try:
            # min_ değeri scalar hale getirilir.
            s_min = float(np.array(getattr(scaler, "min_")).reshape(-1)[0])
            # max_ değeri scalar hale getirilir.
            s_max = float(np.array(getattr(scaler, "max_")).reshape(-1)[0])
            # Okunan min/max döndürülür.
            return s_min, s_max
        except Exception:
            # Custom scaler alanları okunamazsa sklearn formatı denenir.
            pass
    # sklearn MinMaxScaler data_min_ ve data_max_ alanlarına sahiptir.
    if hasattr(scaler, "data_min_") and hasattr(scaler, "data_max_"):
        try:
            # sklearn scaler minimum değeri okunur.
            s_min = float(np.array(getattr(scaler, "data_min_")).reshape(-1)[0])
            # sklearn scaler maximum değeri okunur.
            s_max = float(np.array(getattr(scaler, "data_max_")).reshape(-1)[0])
            # Okunan min/max döndürülür.
            return s_min, s_max
        except Exception:
            # Okuma başarısız olursa None döndürülür.
            pass
    # Hiçbir scaler formatı okunamazsa None döndürülür.
    return None, None

# SCALER UYUMLULUK KONTROLÜ
# Gelen veri eğitim scaler aralığının çok dışına çıkmış mı kontrol eder.
def _check_scaler_compat(arr_min, arr_max, s_min, s_max):
    # Scaler min/max okunamadıysa kontrol yapılamaz; uyumlu kabul edilir.
    if s_min is None or s_max is None:
        return True
    # Scaler range hesaplanır.
    # Eğer min == max ise bölme/0 riskine karşı 1.0 kullanılır.
    s_range = (s_max - s_min) if (s_max != s_min) else 1.0
    # Eğitim aralığının %10'u kadar tolerans verilir.
    tol = 0.10 * abs(s_range)
    # Gelen veri eğitim aralığının toleranslı sınırlarının dışındaysa uyumsuz kabul edilir.
    if (arr_min < (s_min - tol)) or (arr_max > (s_max + tol)):
        return False
    # Aksi halde uyumlu kabul edilir.
    return True

# INFER ENDPOINT
# Ignition/WebDev tarafından gönderilen sensör verileri üzerinde inference yapar.
@router.post("/infer")
def infer(p: InferPayload):
    # Request içindeki opsiyonel meta alanını alır.
    meta = p.meta or {}
    # Backend terminalinde request başlangıcını loglar.
    print("\n================ INFER REQUEST ================")
    print("request_id:", p.request_id)
    print("meta:", meta)
    print("series_count:", len(p.series))
    print("================================================\n")
    # Batch uzunluğu dakika cinsinden alınır.
    # 0 ise batch kullanılmıyor anlamına gelir.
    batch_minutes = int(meta.get("batch_minutes", 0))
    # Batch kayma adımı dakika cinsinden alınır.
    # Verilmezse batch_minutes ile aynı kabul edilir.
    batch_stride_minutes = int(meta.get("batch_stride_minutes", batch_minutes if batch_minutes > 0 else 0))
    # Holdout süresi dakika cinsinden alınır.
    # Forecast için ayrılacak son bölüm olarak kullanılabilir.
    holdout_minutes = int(meta.get("holdout_minutes", 10))
    # UI’dan gelen step_seconds bilgisi alınır.
    # Verilmezse timestamp’lerden hesaplanır.
    step_seconds_meta = meta.get("step_seconds", None)
    # Gelecek forecast serisinin response’a eklenip eklenmeyeceğini belirler.
    include_forecast = bool(meta.get("include_forecast", False))
    # Geriye uyumluluk için duruyor.
    # Bu kodda scaler mismatch hata olarak durdurulmuyor, warning üretiyor.
    allow_scaler_mismatch = bool(meta.get("allow_scaler_mismatch", False))
    # Normalize edilmiş veride 0-1 aralığı dışına çıkma toleransı.
    # Drift/fault gibi durumlarda range flag üretmek için kullanılır.
    range_margin = float(meta.get("range_margin", 0.03))
    # Meta parametrelerini loglar.
    print("META:", {
        "batch_minutes": batch_minutes,
        "batch_stride_minutes": batch_stride_minutes,
        "holdout_minutes": holdout_minutes,
        "step_seconds_meta": step_seconds_meta,
        "include_forecast": include_forecast,
        "allow_scaler_mismatch": allow_scaler_mismatch,
        "anomaly_confirm_points": meta.get("anomaly_confirm_points"),
        "recover_points": meta.get("recover_points"),
        "range_margin": meta.get("range_margin")
    })
    # API response ana objesi.
    out = {"request_id": p.request_id, "results": []}
    # Request içinde bir veya daha fazla seri olabilir.
    # Her tag/cihaz için ayrı inference yapılır.
    for s in p.series:
        # Forecast LSTM modeli yoksa inference yapılamaz.
        if not os.path.exists(mp(s.tag)):
            raise HTTPException(400, f"Forecast Model Yok: {s.tag}")
        # Autoencoder modeli yoksa anomali tespiti yapılamaz.
        if not os.path.exists(mp_ae(s.tag)):
            raise HTTPException(400, f"Autoencoder Model Yok: {s.tag}")
        # Forecast model checkpoint'i yüklenir.
        checkpoint_f = torch.load(mp(s.tag), weights_only=True, map_location="cpu")
        # Autoencoder model checkpoint'i yüklenir.
        checkpoint_ae = torch.load(mp_ae(s.tag), weights_only=True, map_location="cpu")
        # Forecast model parametreleri checkpoint içinden alınır.
        params = checkpoint_f.get("params", checkpoint_f)
        # Autoencoder model parametreleri checkpoint içinden alınır.
        params_ae = checkpoint_ae.get("params", checkpoint_ae)
        # LSTM hidden size parametresi alınır.
        h = params.get("hidden_size", 64)
        # Forecast modeli için look_back alınır.
        look_back = params.get("look_back", 10)
        # Forecast LSTM katman sayısı alınır.
        nl_f = params.get("num_layers", 1)
        # Forecast dropout değeri alınır.
        drop_f = params.get("dropout", 0.0)
        # Autoencoder look_back değeri alınır.
        look_back_ae = params_ae.get("look_back", 10)
        # Autoencoder latent dimension değeri alınır.
        ld_ae = params_ae.get("latent_dim", 32)
        # Autoencoder LSTM katman sayısı alınır.
        nl_ae = params_ae.get("num_layers", 1)
        # Autoencoder dropout değeri alınır.
        drop_ae = params_ae.get("dropout", 0.0)
        # Forecast LSTM modeli aynı parametrelerle yeniden oluşturulur.
        f_model = LSTMModel(hidden_size=h, num_layers=nl_f, dropout=drop_f)
        # Kaydedilmiş ağırlıklar forecast modeline yüklenir.
        f_model.load_state_dict(checkpoint_f["model_state"])
        # Model inference moduna alınır.
        f_model.eval()
        # Autoencoder modeli aynı parametrelerle yeniden oluşturulur.
        ae_model = Autoencoder(look_back_ae, latent_dim=ld_ae, num_layers=nl_ae, dropout=drop_ae)
        # Kaydedilmiş ağırlıklar Autoencoder modeline yüklenir.
        ae_model.load_state_dict(checkpoint_ae["model_state"])
        # Autoencoder inference moduna alınır.
        ae_model.eval()
        # Eğitimde kullanılan scaler yüklenir.
        scaler = joblib.load(sp(s.tag))
        # Yüklenen model bilgileri debug için yazdırılır.
        print("\n---------------- INFER MODEL LOADED ----------------")
        print("tag:", s.tag)
        print("forecast_model_path:", mp(s.tag))
        print("ae_model_path:", mp_ae(s.tag))
        print("scaler_path:", sp(s.tag))
        print("forecast_params:", params)
        print("ae_params:", params_ae)
        print("checkpoint_train_min:", checkpoint_f.get("train_min", None))
        print("checkpoint_train_max:", checkpoint_f.get("train_max", None))
        print("checkpoint_train_mean:", checkpoint_f.get("train_mean", None))
        print("checkpoint_train_std:", checkpoint_f.get("train_std", None))
        print("checkpoint_reference_window_exists:", checkpoint_f.get("reference_window", None) is not None)
        # Eğer checkpoint içinde reference_windows varsa sayısı yazdırılır.
        print(
            "checkpoint_reference_windows_count:",
            len(checkpoint_f.get("reference_windows", []))
            if checkpoint_f.get("reference_windows", None) is not None
            else 0
        )
        # Autoencoder threshold checkpoint içinde varsa yazdırılır.
        print("checkpoint_ae_threshold:", checkpoint_ae.get("threshold", None))
        # Threshold'un hangi kaynaktan geldiği yazdırılır.
        print("checkpoint_threshold_source:", checkpoint_ae.get("threshold_source", None))
        # Scaler min/max bilgisi debug için okunur.
        try:
            dbg_smin, dbg_smax = _get_scaler_minmax(scaler)
            print("scaler_min:", dbg_smin)
            print("scaler_max:", dbg_smax)
        except Exception as e:
            # Scaler debug başarısız olursa loglanır ama sistem durmaz.
            print("scaler debug failed:", str(e))
        print("----------------------------------------------------\n")
        # Gelen sensör değerleri float32 numpy array'e çevrilir.
        arr = np.array(s.values, dtype=np.float32)
        # Veri uzunluğu alınır.
        n = len(arr)
        # En az 30 örnek yoksa inference güvenilir değildir.
        if n < 30:
            raise HTTPException(
                400,
                f"'{s.tag}' için yeterli veri yok. En az 30 geçerli örnek gerekir, gelen örnek sayısı: {n}"
            )
        # NaN veya sonsuz değer varsa model sağlıklı çalışamaz.
        if np.isnan(arr).any() or np.isinf(arr).any():
            raise HTTPException(
                400,
                f"'{s.tag}' içinde NaN veya sonsuz değer var. Model eğitimi iptal edildi."
            )
        # Veri sabitse model/anomali sonucu anlamsız olur.
        if float(np.max(arr) - np.min(arr)) < 1e-6:
            raise HTTPException(
                400,
                f"'{s.tag}' verisi sabit veya anlamsız görünüyor. Null/flat veri ile model eğitilemez."
            )
        # Forecast look_back için yeterli veri olmalı.
        if n <= look_back + 1:
            raise HTTPException(400, f"'{s.tag}' için yeterli veri yok.")
        # Gelen raw verinin minimum değeri.
        arr_min = float(arr.min())
        # Gelen raw verinin maksimum değeri.
        arr_max = float(arr.max())
        # Scaler eğitim min/max aralığı okunur.
        s_min, s_max = _get_scaler_minmax(scaler)
        # Scaler warning başlangıçta yoktur.
        scaler_warning = None
        # Gelen veri eğitim scaler aralığıyla uyumlu mu kontrol edilir.
        scaler_mismatch = not _check_scaler_compat(arr_min, arr_max, s_min, s_max)
        # Eğer veri eğitim aralığının dışına çıktıysa uyarı oluşturulur.
        if scaler_mismatch:
            scaler_warning = (
                f"Incoming data is outside trained scaler range. "
                f"This is allowed during inference and may indicate anomaly/fault. "
                f"ARR_MIN/MAX={arr_min:.4f}/{arr_max:.4f}, "
                f"SCALER_MIN/MAX={s_min}/{s_max}"
            )
            print(f"[WARN] {s.tag} scaler mismatch during infer. Continuing for anomaly detection.")
        # Gelen veri eğitim scaler'ı ile normalize edilir.
        try:
            # Bazı scaler'lar 2D input bekler; bu yüzden reshape(-1, 1) denenir.
            arr_scaled = scaler.transform(arr.reshape(-1, 1)).astype(np.float32).reshape(-1)
        except Exception:
            # Custom scaler 1D input kabul edebilir; fallback olarak 1D transform denenir.
            arr_scaled = scaler.transform(arr).astype(np.float32)
        # Gelen veri istatistikleri debug için yazdırılır.
        print("\n---------------- INFER DATA ----------------")
        print("tag:", s.tag)
        print("n:", n)
        print("raw_min:", float(arr.min()))
        print("raw_max:", float(arr.max()))
        print("raw_mean:", float(arr.mean()))
        print("raw_std:", float(arr.std()))
        print("scaled_min:", float(arr_scaled.min()))
        print("scaled_max:", float(arr_scaled.max()))
        print("scaled_mean:", float(arr_scaled.mean()))
        print("scaled_std:", float(arr_scaled.std()))
        print("first_ts:", s.timestamps[0] if len(s.timestamps) else None)
        print("last_ts:", s.timestamps[-1] if len(s.timestamps) else None)
        print("--------------------------------------------\n")
        # step_seconds UI’dan geldiyse onu kullanır.
        # Gelmediyse timestamp farklarından otomatik hesaplar.
        step_seconds = int(step_seconds_meta) if step_seconds_meta is not None else _infer_step_seconds(s.timestamps)
        # Seri genel bilgilerini loglar.
        print("SERIES INFO:", {
            "tag": s.tag,
            "n": n,
            "look_back": look_back,
            "look_back_ae": look_back_ae,
            "step_seconds": step_seconds,
            "arr_min": arr_min,
            "arr_max": arr_max,
            "scaler_min": s_min,
            "scaler_max": s_max
        })
        # Cihaz bazlı threshold başlangıç değeri.
        threshold_device = 0.0
        # Eğer Autoencoder checkpoint içinde eğitim threshold'u varsa alınır.
        checkpoint_threshold = checkpoint_ae.get("threshold", None)
        # Kayıtlı threshold varsa onu kullanmak en doğru senaryo olur.
        if checkpoint_threshold is not None and float(checkpoint_threshold) > 0:
            threshold_device = float(checkpoint_threshold)
            threshold_source = "checkpoint_train_threshold"
        # Kayıtlı threshold yoksa inference verisi üzerinden threshold hesaplanır.
        elif n > look_back_ae + 1:
            # Tüm seri için Autoencoder input pencereleri oluşturulur.
            X_ae_all = np.array(
                [arr_scaled[i:i + look_back_ae] for i in range(n - look_back_ae)],
                dtype=np.float32
            ).reshape(-1, look_back_ae, 1)
            # Autoencoder ile tüm pencereler yeniden oluşturulur.
            with torch.no_grad():
                recon_all = ae_model(torch.tensor(X_ae_all)).detach().cpu().numpy().reshape(-1, look_back_ae)
            # Reconstruction error hesaplanır.
            # Son noktanın reconstruction değeri gerçek scaled değerle karşılaştırılır.
            errors_all = np.abs(recon_all[:, -1] - arr_scaled[look_back_ae:])
            # Normal aralıkta kalan error değerleri burada tutulur.
            normal_errors = []
            # Tüm error değerleri gezilir.
            for k, err in enumerate(errors_all):
                # Error'ın karşılık geldiği gerçek index.
                idx = k + look_back_ae
                # O andaki normalize değer.
                current_scaled = float(arr_scaled[idx])
                # Autoencoder input penceresi.
                input_window = arr_scaled[idx - look_back_ae:idx]
                # Pencere boşsa atlanır.
                if len(input_window) == 0:
                    continue
                # Hem mevcut değer hem input pencere 0-1 aralığı civarındaysa normal range kabul edilir.
                is_normal_range = (
                    current_scaled >= (0.0 - range_margin)
                    and current_scaled <= (1.0 + range_margin)
                    and float(np.min(input_window)) >= (0.0 - range_margin)
                    and float(np.max(input_window)) <= (1.0 + range_margin)
                )
                # Sadece normal range içindeki error değerleri threshold hesabına dahil edilir.
                if is_normal_range:
                    normal_errors.append(float(err))
            # Yeterli normal error varsa threshold istatistiksel olarak hesaplanır.
            if len(normal_errors) >= 20:
                threshold_device = compute_threshold(np.array(normal_errors, dtype=np.float32))
                threshold_source = "infer_normal_range_only"
            # Yeterli normal error yoksa güvenli default threshold kullanılır.
            else:
                threshold_device = 0.20
                threshold_source = "safe_fallback_0_20"
        # AE threshold hesaplanamayacak kadar az veri varsa fallback kullanılır.
        else:
            threshold_device = 0.20
            threshold_source = "safe_fallback_0_20"
        # Threshold debug bilgileri yazdırılır.
        print("\n---------------- THRESHOLD DEBUG ----------------")
        print("tag:", s.tag)
        print("threshold_device:", threshold_device)
        print("threshold_source:", threshold_source)
        print("checkpoint_ae_threshold:", checkpoint_ae.get("threshold", None))
        print("-------------------------------------------------\n")
        # Batch modu aktifse seri batch'lere bölünerek işlenir.
        if batch_minutes and batch_minutes > 0:
            # Batch kaç örnekten oluşacak hesaplanır.
            # Örneğin 10 dakika ve 5 saniye step -> 120 örnek.
            batch_steps = max(
                look_back + 2,
                round((batch_minutes * 60) / max(1, step_seconds))
            )
            # Batch stride yani kayma adımı hesaplanır.
            stride_steps = (
                max(1, int((batch_stride_minutes * 60) / max(1, step_seconds)))
                if batch_stride_minutes
                else batch_steps
            )
            # Batch konfigürasyonu loglanır.
            print("BATCH CONFIG:", {
                "tag": s.tag,
                "batch_minutes": batch_minutes,
                "batch_steps": batch_steps,
                "stride_steps": stride_steps,
                "n": n,
                "can_make_batch": (batch_steps <= n)
            })
            # Tüm batch sonuçları burada tutulur.
            batches = []
            # İlk batch başlangıç index'i.
            start_idx = 0
            # Yeterli veri olduğu sürece batch oluşturulur.
            while start_idx + batch_steps <= n:
                # Batch bitiş index'i.
                end_idx = start_idx + batch_steps
                # Batch içindeki normalize edilmiş değerler.
                win_scaled = arr_scaled[start_idx:end_idx]
                # Batch içindeki raw sensör değerleri.
                win_vals = arr[start_idx:end_idx]
                # Batch içindeki timestamp değerleri.
                win_ts = s.timestamps[start_idx:end_idx]
                # Batch örnek sayısı.
                win_n = len(win_scaled)
                # Batch debug bilgileri loglanır.
                print("\n================ BATCH DEBUG ================")
                print("tag:", s.tag)
                print("start_idx:", start_idx)
                print("end_idx:", end_idx)
                print("win_n:", win_n)
                print("win_raw_min:", float(win_vals.min()))
                print("win_raw_max:", float(win_vals.max()))
                print("win_raw_mean:", float(win_vals.mean()))
                print("win_scaled_min:", float(win_scaled.min()))
                print("win_scaled_max:", float(win_scaled.max()))
                print("win_scaled_mean:", float(win_scaled.mean()))
                print("batch_start_ts:", win_ts[0])
                print("batch_end_ts:", win_ts[-1])
                print("=============================================\n")
                # Batch pencere bilgisi loglanır.
                print("BATCH WINDOW:", {
                    "tag": s.tag,
                    "start_idx": start_idx,
                    "end_idx": start_idx + batch_steps,
                    "n": n
                })
                # Nokta bazlı anomaliler burada tutulur.
                anomalies = []
                # Event bazlı anomaliler burada tutulur.
                # Ardışık anomaliler tek event başlangıcı olarak sayılabilir.
                anomaly_events = []
                # Grafik için anomaly score zaman serisi burada tutulur.
                anomaly_score = []
                # Grafik için threshold çizgisi burada tutulur.
                threshold_series = []
                # Index -> raw reconstruction error map.
                error_by_idx = {}
                # Index -> karar için kullanılan score map.
                decision_score_by_idx = {}
                # Index -> range flag map.
                range_flag_by_idx = {}
                # Batch içinde AE hesaplanabilecek kadar veri varsa:
                if win_n > look_back_ae + 1:
                    # Batch için Autoencoder pencereleri oluşturulur.
                    X_ae = np.array([
                        win_scaled[i:i + look_back_ae]
                        for i in range(win_n - look_back_ae)
                    ], dtype=np.float32).reshape(-1, look_back_ae, 1)
                    # Autoencoder reconstruction üretir.
                    with torch.no_grad():
                        recon = ae_model(torch.tensor(X_ae)).numpy().reshape(-1, look_back_ae)
                    # Reconstruction error hesaplanır.
                    errors = np.abs(recon[:, -1] - win_scaled[look_back_ae:])
                    # Bu batch için kullanılacak threshold cihaz bazlı threshold'dur.
                    threshold = threshold_device
                    # Önceki noktanın anomali olup olmadığını tutar.
                    # Event başlangıcı yakalamak için kullanılır.
                    prev_is_anomaly = False
                    # Her reconstruction error gezilir.
                    for i, err in enumerate(errors):
                        # Error'ın batch içindeki gerçek index'i.
                        idx = i + look_back_ae
                        # İlgili timestamp.
                        t_raw = win_ts[idx]
                        # İlgili normalize edilmiş değer.
                        current_scaled = float(win_scaled[idx])
                        # Forecast look_back'e göre input pencere başlangıcı.
                        input_start = max(0, idx - look_back)
                        # Karar için kullanılan input pencere.
                        input_window = win_scaled[input_start:idx]
                        # Input pencere varsa min/max alınır.
                        if len(input_window) > 0:
                            window_min = float(np.min(input_window))
                            window_max = float(np.max(input_window))
                        # Input pencere yoksa mevcut değer kullanılır.
                        else:
                            window_min = current_scaled
                            window_max = current_scaled
                        # Mevcut değer normalize 0-1 aralığının dışına çıktı mı?
                        current_range_flag = (
                            current_scaled < (0.0 - range_margin)
                            or current_scaled > (1.0 + range_margin)
                        )
                        # Input pencere içinde 0-1 aralığının dışına çıkan değer var mı?
                        window_range_flag = (
                            window_min < (0.0 - range_margin)
                            or window_max > (1.0 + range_margin)
                        )
                        # Mevcut nokta veya pencere range dışıysa range_flag true olur.
                        range_flag = current_range_flag or window_range_flag
                        # Eğer range dışı bir durum varsa decision_score threshold'u kesin aşacak şekilde yükseltilir.
                        # Böylece scaler aralığı dışına çıkan fault/drift durumları kaçırılmaz.
                        if range_flag and threshold > 0:
                            decision_score = max(float(err), float(threshold) * 1.25)
                        # Range dışı değilse karar skoru raw reconstruction error olur.
                        else:
                            decision_score = float(err)
                        # Raw reconstruction error index'e göre kaydedilir.
                        error_by_idx[idx] = float(err)
                        # Karar skoru index'e göre kaydedilir.
                        decision_score_by_idx[idx] = float(decision_score)
                        # Range flag index'e göre kaydedilir.
                        range_flag_by_idx[idx] = bool(range_flag)
                        # Anomaly score grafiği için nokta eklenir.
                        anomaly_score.append({
                            "t": int(_parse_dt(t_raw).timestamp() * 1000),
                            "value": round(float(decision_score), 6),
                            "raw_error": round(float(err), 6),
                            "range_flag": bool(range_flag),
                            "current_range_flag": bool(current_range_flag),
                            "window_range_flag": bool(window_range_flag)
                        })
                        # Karar skoru threshold'u aşarsa anomali kabul edilir.
                        is_anomaly = decision_score > threshold
                        # Nokta anomaliyse tabloya ve sonuçlara eklenir.
                        if is_anomaly:
                            anomalies.append({
                                "t": t_raw,
                                "value": round(float(win_vals[idx]), 4),
                                "error": round(float(decision_score), 4),
                                "raw_error": round(float(err), 4),
                                "range_flag": bool(range_flag),
                                "reason": "anomalous_point"
                            })
                        # Eğer bu nokta anomaliyse ve önceki nokta anomali değilse yeni event başlangıcıdır.
                        if is_anomaly and not prev_is_anomaly:
                            anomaly_events.append({
                                "t": t_raw,
                                "value": round(float(win_vals[idx]), 4),
                                "error": round(float(decision_score), 4),
                                "raw_error": round(float(err), 4),
                                "range_flag": bool(range_flag),
                                "reason": "event_start"
                            })
                        # Bir sonraki iterasyon için önceki durum güncellenir.
                        prev_is_anomaly = is_anomaly
                    # Eğer anomaly score oluştuysa threshold çizgisi iki noktayla oluşturulur.
                    if anomaly_score:
                        threshold_series = [
                            {
                                "t": anomaly_score[0]["t"],
                                "value": round(float(threshold), 6)
                            },
                            {
                                "t": anomaly_score[-1]["t"],
                                "value": round(float(threshold), 6)
                            }
                        ]
                # Batch AE için yeterli değilse threshold ve errors default kalır.
                else:
                    threshold = 0.0
                    errors = []
                    anomaly_events = []
                # Holdout kaç step olacak hesaplanır.
                holdout_steps = max(
                    1,
                    min(
                        int((holdout_minutes * 60) / max(1, step_seconds)),
                        max(1, win_n - (look_back + 1))
                    )
                )
                # Holdout dışındaki observed bölümün bitiş index'i.
                obs_rel_end = win_n - holdout_steps
                # Observed normalize veri.
                obs_scaled = win_scaled[:obs_rel_end]
                # Observed raw veri.
                obs_vals = win_vals[:obs_rel_end]
                # Observed timestamp'ler.
                obs_ts = win_ts[:obs_rel_end]
                # Actual data grafiği için raw seri hazırlanır.
                actual = [{
                    "t": int(_parse_dt(t).timestamp() * 1000),
                    "value": round(float(v), 4)
                } for t, v in zip(win_ts, win_vals)]
                # Predicted data grafiği için liste.
                predicted = []
                # Normalize predicted değerler burada tutulur.
                preds_scaled = []
                # Predicted timestamp'ler burada tutulur.
                preds_ts = []
                # Eğitim sırasında kaydedilmiş normal referans pencere alınır.
                reference_window = checkpoint_f.get("reference_window", None)
                # Reference window varsa fallback seed olarak kullanılır.
                if reference_window is not None and len(reference_window) >= look_back:
                    fallback_seed = list(reference_window[-look_back:])
                # Yoksa batch'in ilk look_back değeri fallback seed olur.
                else:
                    fallback_seed = win_scaled[:look_back].astype(np.float32).tolist()
                # Kaç ardışık contaminated nokta sonrası anomaly mode'a geçilecek.
                anomaly_confirm_points = int(meta.get("anomaly_confirm_points", 3))
                # Kaç temiz nokta sonrası normal mode'a dönülecek.
                recover_points = int(meta.get("recover_points", 6))
                # Prediction başlangıçta normal mode'dadır.
                anomaly_mode = False
                # Ardışık contaminated nokta sayacı.
                over_count = 0
                # Ardışık temiz nokta sayacı.
                under_count = 0
                # Son normal pencere burada tutulur.
                last_normal_window = list(fallback_seed)
                # Recursive anomaly mode prediction penceresi.
                pred_window = list(fallback_seed)
                # Prediction üretmek için yeterli veri varsa:
                if win_n > look_back:
                    # Prediction sırasında gradient hesaplanmaz.
                    with torch.no_grad():
                        # look_back'ten batch sonuna kadar her nokta için tahmin üretilir.
                        for i in range(look_back, win_n):
                            # Bu index için karar skoru alınır.
                            # Yoksa raw error veya 0 kullanılır.
                            current_err = decision_score_by_idx.get(i, error_by_idx.get(i, 0.0))
                            # Mevcut normalize değer.
                            current_scaled = float(win_scaled[i])
                            # Forecast input penceresi.
                            input_window = win_scaled[i - look_back:i]
                            # Mevcut değer range dışında mı?
                            current_range_flag = (
                                current_scaled < (0.0 - range_margin)
                                or current_scaled > (1.0 + range_margin)
                            )
                            # Input pencere range dışında değer içeriyor mu?
                            window_range_flag = (
                                float(np.min(input_window)) < (0.0 - range_margin)
                                or float(np.max(input_window)) > (1.0 + range_margin)
                            )
                            # Nokta contaminated kabul edilir mi?
                            # Threshold üstü error veya range dışı durum varsa contaminated olur.
                            contaminated = (
                                current_err > threshold
                                or current_range_flag
                                or window_range_flag
                            )
                            # Her 20 noktada veya contaminated olduğunda karar loglanır.
                            if i % 20 == 0 or contaminated:
                                print("PREDICT DECISION:", {
                                    "tag": s.tag,
                                    "i": i,
                                    "ts": win_ts[i],
                                    "raw": round(float(win_vals[i]), 4),
                                    "scaled": round(float(current_scaled), 6),
                                    "current_err": round(float(current_err), 6),
                                    "threshold": round(float(threshold), 6),
                                    "over_threshold": bool(current_err > threshold),
                                    "current_range_flag": bool(current_range_flag),
                                    "window_range_flag": bool(window_range_flag),
                                    "contaminated": bool(contaminated),
                                    "anomaly_mode": bool(anomaly_mode),
                                    "over_count": over_count,
                                    "under_count": under_count
                                })
                            # Contaminated ise over_count artırılır.
                            if contaminated:
                                over_count += 1
                                under_count = 0
                            # Temiz ise under_count artırılır.
                            else:
                                under_count += 1
                                over_count = 0
                            # Normal moddan anomaly moda geçiş.
                            # Amaç: fault/drift başladığında prediction'ın actual fault'u takip etmemesi.
                            if (not anomaly_mode) and (over_count >= anomaly_confirm_points):
                                anomaly_mode = True
                                # Recursive prediction, son normal pencereden başlatılır.
                                pred_window = list(last_normal_window)
                            # Anomaly moddayken yeterince temiz nokta gelirse normale dönüş.
                            elif anomaly_mode and (under_count >= recover_points):
                                anomaly_mode = False
                            # Normal mode prediction.
                            if not anomaly_mode:
                                # Normal modda model gerçek son look_back veriyi kullanır.
                                actual_window = win_scaled[i - look_back:i].astype(np.float32).tolist()
                                # Eğer nokta contaminated değilse son normal pencere güncellenir.
                                if not contaminated:
                                    last_normal_window = list(actual_window)
                                # LSTM input tensor oluşturulur.
                                w = torch.tensor(
                                    actual_window,
                                    dtype=torch.float32
                                ).unsqueeze(0).unsqueeze(-1)
                                # Forecast modelinden normalize tahmin alınır.
                                yhat_scaled = float(f_model(w).item())
                            # Anomaly mode prediction.
                            else:
                                # Anomaly modda actual fault verisi kullanılmaz.
                                # Kendi tahminlerinden oluşan recursive pencere kullanılır.
                                w = torch.tensor(
                                    pred_window[-look_back:],
                                    dtype=torch.float32
                                ).unsqueeze(0).unsqueeze(-1)
                                # Forecast modelinden normalize tahmin alınır.
                                yhat_scaled = float(f_model(w).item())
                                # Tahmin recursive pencereye eklenir.
                                pred_window.append(yhat_scaled)
                            # Normalize tahmin listeye eklenir.
                            preds_scaled.append(yhat_scaled)
                            # Tahminin timestamp'i eklenir.
                            preds_ts.append(win_ts[i])
                    # Normalize predicted değerler raw ölçeğe geri çevrilir.
                    try:
                        preds_vals = scaler.inverse_transform(
                            np.array(preds_scaled, dtype=np.float32).reshape(-1, 1)
                        ).flatten()
                    except Exception:
                        # Custom scaler 1D input bekliyorsa fallback yapılır.
                        preds_vals = scaler.inverse_transform(
                            np.array(preds_scaled, dtype=np.float32)
                        )
                    # Predicted seri response formatına çevrilir.
                    for t, v in zip(preds_ts, preds_vals):
                        predicted.append({
                            "t": int(_parse_dt(t).timestamp() * 1000),
                            "value": round(float(v), 4)
                        })
                # Forecast listesi başlangıçta boştur.
                forecast = []
                # include_forecast true ise gelecek değer forecast'i üretilir.
                if include_forecast:
                    # Normalize forecast değerleri burada tutulur.
                    forecast_preds = []
                    # Observed veri forecast için yeterliyse:
                    if len(obs_scaled) >= look_back:
                        # Forecast başlangıç penceresi observed kısmın son look_back değeridir.
                        f_window = obs_scaled[-look_back:].tolist()
                        # Forecast uzunluğu holdout_steps kadar olur.
                        forecast_len = holdout_steps
                        # Forecast sırasında gradient hesaplanmaz.
                        with torch.no_grad():
                            # Recursive şekilde geleceğe doğru tahmin üretilir.
                            for _ in range(forecast_len):
                                # LSTM input tensor oluşturulur.
                                w = torch.tensor(
                                    f_window[-look_back:],
                                    dtype=torch.float32
                                ).unsqueeze(0).unsqueeze(-1)
                                # Bir sonraki normalize değer tahmin edilir.
                                next_scaled = f_model(w).item()
                                # Tahmin pencereye eklenir.
                                f_window.append(next_scaled)
                                # Tahmin forecast listesine eklenir.
                                forecast_preds.append(next_scaled)
                        # Forecast değerleri raw ölçeğe geri çevrilir.
                        try:
                            forecast_values = scaler.inverse_transform(
                                np.array(forecast_preds, dtype=np.float32).reshape(-1, 1)
                            ).flatten()
                        except Exception:
                            # Custom scaler 1D bekliyorsa fallback yapılır.
                            forecast_values = scaler.inverse_transform(
                                np.array(forecast_preds, dtype=np.float32)
                            )
                        # Gelecek timestamp'ler üretilir.
                        future_ts = generate_future_timestamps(obs_ts[-1], holdout_steps, step_seconds)
                        # Forecast çizgisinin actual ile bağlanması için anchor noktası eklenir.
                        anchor = int(_parse_dt(obs_ts[-1]).timestamp() * 1000)
                        # Anchor noktası observed son değerdir.
                        forecast.append({
                            "t": anchor,
                            "forecast_value": round(float(obs_vals[-1]), 4)
                        })
                        # Gelecek tahmin noktaları response’a eklenir.
                        for t, v in zip(future_ts, forecast_values):
                            forecast.append({
                                "t": int(_parse_dt(t).timestamp() * 1000),
                                "forecast_value": round(float(v), 4)
                            })
                # Batch sonucu terminale yazdırılır.
                print("BATCH RESULT:", {
                    "tag": s.tag,
                    "anomaly_count": len(anomalies),
                    "anomaly_event_count": len(anomaly_events),
                    "actual_count": len(actual),
                    "predicted_count": len(predicted),
                    "forecast_count": len(forecast),
                    "anomaly_score_count": len(anomaly_score)
                })
                # Batch sonucu response listesine eklenir.
                batches.append({
                    "start": win_ts[0],
                    "end": win_ts[-1],
                    "actual": actual,
                    "predicted": predicted,
                    "forecast": forecast,
                    "anomalies": anomalies,
                    "anomaly_events": anomaly_events,
                    "anomaly_score": anomaly_score,
                    "threshold_series": threshold_series,
                    "summary": {
                        "anomaly_count": len(anomalies),
                        "anomaly_event_count": len(anomaly_events),
                        "threshold_used": round(float(threshold), 4),
                        "threshold_source": threshold_source,
                        "holdout_minutes": holdout_minutes,
                        "step_seconds": step_seconds,
                        "batch_minutes": batch_minutes,
                        "scaler_warning": scaler_warning
                    }
                })
                # Bir sonraki batch'e geçilir.
                start_idx += stride_steps
            # Tüm batch'lerdeki toplam nokta bazlı anomali sayısı.
            total_anoms = sum(b["summary"]["anomaly_count"] for b in batches)
            # Tüm batch'lerdeki toplam event bazlı anomali sayısı.
            total_event_anoms = sum(b["summary"]["anomaly_event_count"] for b in batches)
            # Tag bazlı final sonuç loglanır.
            print("FINAL TAG RESULT:", {
                "tag": s.tag,
                "batches_count": len(batches),
                "total_anomaly_count": total_anoms,
                "total_anomaly_event_count": total_event_anoms
            })
            # Bu tag için tüm inference sonucu ana response’a eklenir.
            out["results"].append({
                "tag": s.tag,
                "batches": batches,
                "summary": {
                    "batches_count": len(batches),
                    "total_anomaly_count": total_anoms,
                    "total_anomaly_event_count": total_event_anoms,
                    "batch_minutes": batch_minutes,
                    "batch_stride_minutes": batch_stride_minutes if batch_stride_minutes else batch_minutes,
                    "holdout_minutes": holdout_minutes,
                    "step_seconds": step_seconds,
                    "threshold_used": round(float(threshold_device), 4),
                    "threshold_source": threshold_source,
                    "scaler_warning": scaler_warning
                }
            })
    # Tüm tag sonuçlarını içeren response döndürülür.
    return out