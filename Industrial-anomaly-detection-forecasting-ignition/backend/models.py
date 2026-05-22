# PyTorch ana kütüphanesi.
# Tensor işlemleri, model kaydetme/yükleme ve GPU kullanımı için gerekli.
import torch
# PyTorch neural network modülü.
# nn.Module, LSTM, Linear, Dropout, MSELoss gibi katmanlar buradan gelir.
import torch.nn as nn
# DataLoader ve TensorDataset, eğitim verisini batch'lere ayırmak için kullanılır.
from torch.utils.data import DataLoader, TensorDataset
# NumPy, veriyi array formatına çevirmek ve pencereleme işlemleri için kullanılır.
import numpy as np

# CUDA varsa modeli GPU üzerinde, yoksa CPU üzerinde çalıştırır.
# Böylece kod hem GPU'lu hem GPU'suz bilgisayarda çalışabilir.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LSTM FORECAST MODEL
# Bu model zaman serisi tahmini için kullanılır.
# Geçmiş look_back kadar sensör değerine bakıp bir sonraki değeri tahmin eder.
class LSTMModel(nn.Module):
   # Modelin katmanları burada tanımlanır.
   # hidden_size: LSTM'in iç temsil boyutu.
   # num_layers: kaç LSTM katmanı kullanılacağı.
   # dropout: overfitting azaltmak için kullanılan dropout oranı.
   def __init__(self, hidden_size=64, num_layers=1, dropout=0.0):
       # nn.Module sınıfının kurucu fonksiyonunu çağırır.
       super().__init__()
       # LSTM katmanı.
       # input_size=1 çünkü her timestamp için tek sensör değeri geliyor.
       # hidden_size modelin öğrenme kapasitesini belirler.
       # num_layers birden fazla LSTM katmanı kullanılmasını sağlar.
       # batch_first=True sayesinde input shape şu olur: (batch, sequence, feature).
       # PyTorch LSTM dropout'u sadece num_layers > 1 ise aktif kullanır.
       self.lstm = nn.LSTM(
           input_size=1,
           hidden_size=hidden_size,
           num_layers=num_layers,
           batch_first=True,
           dropout=dropout if num_layers > 1 else 0.0
       )

       # Ek dropout katmanı.
       # LSTM çıktısından sonra uygulanır.
       # Amaç modelin ezberlemesini azaltmak.
       self.drop = nn.Dropout(dropout)

       # Fully connected katman.
       # LSTM'in son hidden output'unu tek bir tahmin değerine dönüştürür.
       self.fc = nn.Linear(hidden_size, 1)

   # Modelin ileri yayılım fonksiyonu.
   # x shape: (batch_size, look_back, 1)
   def forward(self, x):
       # LSTM çıktısını üretir.
       # out shape: (batch_size, look_back, hidden_size)
       # ikinci çıktı hidden/cell state'tir; burada kullanılmadığı için "_" ile alınır.
       out, _ = self.lstm(x)
       # Sadece son zaman adımının çıktısı alınır: out[:, -1, :]
       # Çünkü bir sonraki değeri tahmin ederken tüm pencerenin son temsilini kullanıyoruz.
       # Dropout uygulanır ve fc katmanı ile tek değere dönüştürülür.
       return self.fc(self.drop(out[:, -1, :]))

# LSTM AUTOENCODER MODEL
# Bu model anomali tespiti için kullanılır.
# Normal veriyi yeniden oluşturmayı öğrenir.
# Eğer gelen veri normalden farklıysa reconstruction error yükselir.
class Autoencoder(nn.Module):
   # Autoencoder modelinin katmanları burada tanımlanır.
   # look_back: pencere uzunluğu.
   # latent_dim: encoder'ın sıkıştırılmış temsil boyutu.
   # num_layers: LSTM katman sayısı.
   # dropout: overfitting azaltmak için kullanılan oran.
   def __init__(self, look_back, latent_dim=32, num_layers=1, dropout=0.0):

       # nn.Module sınıfının kurucu fonksiyonunu çağırır.
       super().__init__()
       # Encoder LSTM.
       # Giriş olarak sensör penceresini alır.
       # Veriyi latent_dim boyutunda temsil etmeyi öğrenir.
       self.encoder = nn.LSTM(
           1,
           latent_dim,
           num_layers=num_layers,
           batch_first=True,
           dropout=dropout if num_layers > 1 else 0.0
       )
       # Decoder LSTM.
       # Encoder çıktısını tekrar orijinal veri boyutuna yaklaştırmaya çalışır.
       # Burada output_size tekrar 1 olur çünkü sensör değeri tek boyutludur.
       self.decoder = nn.LSTM(
           latent_dim,
           1,
           num_layers=num_layers,
           batch_first=True,
           dropout=dropout if num_layers > 1 else 0.0
       )
       # Encoder çıktısına dropout uygular.
       # Modelin normal veriyi ezberlemesini bir miktar azaltır.
       self.drop = nn.Dropout(dropout)
   # Autoencoder ileri yayılım fonksiyonu.
   # x shape: (batch_size, look_back, 1)
   def forward(self, x):
       # Encoder giriş penceresini latent temsile dönüştürür.
       # encoded shape: (batch_size, look_back, latent_dim)
       encoded, _ = self.encoder(x)
       # Dropout uygulanmış encoded veri decoder'a verilir.
       # Decoder aynı zaman serisini yeniden üretmeye çalışır.
       # decoded shape: (batch_size, look_back, 1)
       decoded, _ = self.decoder(self.drop(encoded))
       # Yeniden oluşturulmuş seri döndürülür.
       return decoded

# HYPERPARAMETER SEARCH SPACE
# Bu fonksiyon verinin uzunluğuna ve GPU durumuna göre denenecek parametreleri belirler.
def get_search_space(n):
   # Eğer GPU varsa daha geniş parametre kombinasyonları denenebilir.
   # Çünkü GPU eğitim süresini hızlandırır.
   if torch.cuda.is_available():
       # GPU için daha geniş arama alanı.
       return {
           # LSTM hidden size seçenekleri.
           # Büyük değer model kapasitesini artırır ama eğitim süresini de artırır.
           "hidden_sizes":  [32, 64, 128, 256],
           # Autoencoder latent representation boyutu seçenekleri.
           "latent_dims":   [16, 32, 64, 128],
           # LSTM katman sayısı seçenekleri.
           "num_layers":    [1, 2],
           # Veri uzunluğuna göre look_back seçenekleri.
           # n > 60 ise daha uzun geçmiş pencereleri denenebilir.
           "look_backs":    [5, 10, 20, 30] if n > 60 else ([5, 10, 20] if n > 30 else [5, 10]),
           # Öğrenme oranı seçenekleri.
           # Küçük lr daha stabil, büyük lr daha hızlı ama riskli olabilir.
           "lrs":           [0.01, 0.001, 0.0005, 0.0001],
           # Batch size seçenekleri.
           # Büyük batch hızlı olabilir ama her zaman daha iyi sonuç vermez.
           "batch_sizes":   [16, 32, 64],
           # Dropout seçenekleri.
           # Overfitting azaltmak için denenir.
           "dropouts":      [0.0, 0.2, 0.3],
       }
   # GPU yoksa CPU kullanılır.
   # CPU'da eğitim yavaş olacağı için arama alanı daha küçük tutulur.
   else:
       # CPU için daha hafif arama alanı.
       return {
           # CPU'da daha küçük hidden size değerleri denenir.
           "hidden_sizes":  [32, 64],
           # Autoencoder için daha küçük latent dim seçenekleri.
           "latent_dims":   [16, 32],
           # CPU için tek katman kullanılır.
           "num_layers":    [1],
           # Veri uzunluğuna göre kısa look_back seçenekleri.
           "look_backs":    [5, 10] if n > 30 else [5],
           # Daha stabil öğrenme oranları.
           "lrs":           [0.001, 0.0005],
           # CPU için batch size sabit ve küçük tutulur.
           "batch_sizes":   [16],
           # Dropout seçenekleri.
           "dropouts":      [0.0, 0.2],
       }

# SLIDING WINDOW OLUŞTURMA
# Zaman serisini modelin eğitebileceği X-y çiftlerine çevirir.
# Örnek:
# arr = [10, 11, 12, 13], look_back=2
# X = [[10, 11], [11, 12]]
# y = [12, 13]
def _make_windows(arr, look_back):
   # Girdi pencereleri burada tutulur.
   X = []
   # Her pencerenin tahmin hedefi burada tutulur.
   y = []

   # look_back uzunluğunda kayan pencere oluşturulur.
   # Son look_back'ten sonraki değer hedef olarak kullanılır.
   for i in range(len(arr) - look_back):
       # i noktasından başlayarak look_back kadar geçmiş değer alınır.
       X.append(arr[i: i + look_back])
       # Pencerenin hemen sonrasındaki değer hedef olarak alınır.
       y.append(arr[i + look_back])
   # X numpy array'e çevrilir.
   # Shape: (num_samples, look_back, 1)
   # 1 burada tek sensör feature'ını temsil eder.
   X = np.array(X, dtype=np.float32).reshape(-1, look_back, 1)
   # y numpy array'e çevrilir.
   # Shape: (num_samples, 1)
   y = np.array(y, dtype=np.float32).reshape(-1, 1)
   # PyTorch tensor formatında döndürülür.
   return torch.tensor(X), torch.tensor(y)

# TRAIN + VALIDATION LOOP
# Bu fonksiyon tek bir modeli eğitir ve validation loss değerini hesaplar.
# ae=False ise LSTM forecast modeli eğitilir.
# ae=True ise Autoencoder eğitilir.
def _train_eval(model, loader, X_val, y_val, epochs, lr, device, ae=False):
   # Eğitimin başladığını loglar.
   print(f"  Training start | lr={lr} | ae={ae} | epochs={epochs}")
   # Modeli CPU veya GPU cihazına taşır.
   model.to(device)
   # Adam optimizer kullanılır.
   # Model parametrelerini loss'a göre günceller.
   optimizer = torch.optim.Adam(model.parameters(), lr=lr)
   # Validation loss iyileşmezse learning rate'i azaltır.
   # patience=3: 3 epoch boyunca gelişme olmazsa lr düşürülür.
   # factor=0.5: learning rate yarıya indirilir.
   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
       optimizer,
       patience=3,
       factor=0.5
   )
   # Mean Squared Error loss.
   # Forecast için tahmin-gerçek farkını,
   # Autoencoder için reconstruction hatasını ölçer.
   criterion = nn.MSELoss()
   # En iyi validation loss başlangıçta sonsuz atanır.
   best_val_loss = float("inf")
   # Early stopping patience değeri.
   # Validation iyileşmezse en fazla 5 epoch beklenir.
   patience = 5
   # Kaç epoch'tur iyileşme olmadığını sayar.
   trigger_times = 0

   # Belirlenen epoch sayısı kadar eğitim yapılır.
   for ep in range(epochs):
       # Model training moduna alınır.
       # Dropout gibi katmanlar bu modda aktif olur.
       model.train()
       # Epoch boyunca toplam eğitim loss'u burada birikir.
       epoch_loss = 0.0
       # DataLoader batch batch veri döndürür.
       for batch in loader:
           # Batch'in X kısmı cihaza taşınır.
           X_b = batch[0].to(device)
           # Autoencoder için hedef, girişin kendisidir.
           # Çünkü autoencoder input'u tekrar üretmeye çalışır.
           # LSTM forecast için hedef batch[1]'dir.
           y_b = X_b if ae else batch[1].to(device)
           # Önceki gradient değerleri sıfırlanır
           optimizer.zero_grad()
           # Model tahmin üretir ve loss hesaplanır.
           loss = criterion(model(X_b), y_b)
           # Loss'a göre gradient hesaplanır.
           loss.backward()
           # Gradient clipping uygulanır.
           # Ani büyük gradient patlamalarını engeller.
           torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
           # Optimizer model parametrelerini günceller.
           optimizer.step()
           # Batch loss epoch toplamına eklenir.
           epoch_loss += loss.item()
       # Epoch eğitim kaybı loglanır.
       print(f"   Epoch {ep+1}/{epochs} | Train Loss={epoch_loss:.6f}")
       # Model evaluation moduna alınır.
       # Dropout gibi katmanlar bu modda pasif olur.
       model.eval()
       # Validation sırasında gradient hesaplanmaz.
       # Bu hem hız kazandırır hem memory kullanımını azaltır.
       with torch.no_grad():
           # Validation set boşsa değerlendirme yapılamaz.
           if len(X_val) == 0:
               # Validation olmadığı bilgisi loglanır.
               print("  No validation set, skipping eval.")
               # Model döndürülür, loss sonsuz verilir.
               return model, float("inf")
           # Validation input cihaza taşınır.
           X_val_t = X_val.to(device)
           # Autoencoder için validation hedefi input'un kendisidir.
           # Forecast için y_val kullanılır.
           y_val_t = X_val_t if ae else y_val.to(device)
           # Validation loss hesaplanır.
           val_loss = criterion(model(X_val_t), y_val_t).item()
       # Validation loss loglanır.
       print(f"       Val Loss={val_loss:.6f}")
       # Eğer validation loss daha iyi olduysa:
       if val_loss < best_val_loss:
           # En iyi loss güncellenir.
           best_val_loss = val_loss
           # Early stopping sayacı sıfırlanır.
           trigger_times = 0
       # Eğer validation loss iyileşmediyse:
       else:
           # İyileşmeyen epoch sayacı artırılır.
           trigger_times += 1
           # Early stopping durumu loglanır.
           print(f"       No improvement. patience [{trigger_times}/{patience}]")
           # Eğer patience sınırına ulaşıldıysa eğitim durdurulur.
           if trigger_times >= patience:
               # Early stopping bilgisi loglanır.
               print(" Early Stopping Triggered!")
               # Eğitim döngüsünden çıkılır.
               break
       # Scheduler validation loss'a göre learning rate'i günceller.
       scheduler.step(val_loss)
   # Eğitilmiş model ve en iyi validation loss döndürülür.
   return model, best_val_loss

# HYPERPARAMETER OPTIMIZATION
# Bu fonksiyon LSTM veya Autoencoder için farklı parametre kombinasyonlarını dener.
# En düşük validation loss veren modeli ve parametreleri döndürür.
def hyperopt(arr_scaled, n, model_type, epochs, device=DEVICE):
   # Hyperparameter search başlangıcı loglanır.
   print(f"\n Hyperopt started | model={model_type} | data_len={n}\n")
   # Veri uzunluğu ve cihaz durumuna göre search space alınır.
   sp = get_search_space(n)
   # Denenecek parametre aralığı loglanır.
   print(f" Search space: {sp}\n")
   # En iyi loss başlangıçta sonsuzdur.
   best_loss = float("inf")
   # En iyi model başlangıçta yoktur.
   best_model = None
   # En iyi parametreler burada tutulur.
   best_params = {}
   # Farklı look_back değerleri denenir.
   for lb in sp["look_backs"]:
       # Denenen look_back loglanır.
       print(f"\n Trying look_back = {lb}")
       # Ölçeklenmiş zaman serisinden pencere verisi oluşturulur.
       X_t, y_t = _make_windows(arr_scaled, lb)
       # Çok az örnek varsa bu kombinasyon atlanır.
       if len(X_t) < 4:
           # Yetersiz örnek bilgisi loglanır.
           print(" Too few samples, skipping.")
           # Bir sonraki look_back değerine geçilir.
           continue
       # Verinin %80'i train, %20'si validation yapılır.
       split = max(1, int(len(X_t) * 0.8))
       # Eğitim inputları.
       X_tr = X_t[:split]
       # Validation inputları.
       X_val = X_t[split:]
       # Eğitim hedefleri.
       y_tr = y_t[:split]
       # Validation hedefleri.
       y_val = y_t[split:]
       # Learning rate seçenekleri denenir.
       for lr in sp["lrs"]:
           # Batch size seçenekleri denenir.
           for bs in sp["batch_sizes"]:
               # Eğer model_type "lstm" ise forecast modeli eğitilir.
               if model_type == "lstm":
                   # Eğitim verisi TensorDataset'e dönüştürülür.
                   # LSTM forecast için hem X hem y gerekir.
                   loader = DataLoader(
                       TensorDataset(X_tr, y_tr),
                       batch_size=bs,
                       shuffle=True
                   )
                   # Hidden size seçenekleri denenir.
                   for hs in sp["hidden_sizes"]:
                       # Num layers seçenekleri denenir.
                       for nl in sp["num_layers"]:
                           # Dropout seçenekleri denenir.
                           for dr in sp["dropouts"]:
                               # Denenen LSTM parametreleri loglanır.
                               print(
                                   f"\n LSTM | lb={lb} | hs={hs} | nl={nl} | dr={dr} | lr={lr} | bs={bs}"
                               )
                               # Belirlenen parametrelerle LSTM modeli oluşturulur.
                               m = LSTMModel(
                                   hidden_size=hs,
                                   num_layers=nl,
                                   dropout=dr
                               )
                               # Model eğitilir ve validation loss alınır.
                               m, loss = _train_eval(
                                   m,
                                   loader,
                                   X_val,
                                   y_val,
                                   epochs,
                                   lr,
                                   device,
                                   ae=False
                               )
                               # Bu kombinasyonun loss değeri loglanır.
                               print(f" Model Loss = {loss:.6f}")
                               # Eğer bu model şimdiye kadarki en iyi modelden daha iyiyse:
                               if loss < best_loss:
                                   # Yeni en iyi model bilgisi loglanır.
                                   print(" New BEST model found!")
                                   # En iyi loss güncellenir.
                                   best_loss = loss
                                   # En iyi model güncellenir.
                                   best_model = m
                                   # En iyi parametreler kaydedilir.
                                   best_params = {
                                       "model_type": "lstm",
                                       "look_back": lb,
                                       "hidden_size": hs,
                                       "num_layers": nl,
                                       "dropout": dr,
                                       "lr": lr,
                                       "batch_size": bs
                                   }
               # Eğer model_type "lstm" değilse Autoencoder eğitilir.
               else:
                   # Autoencoder için sadece X gerekir.
                   # Çünkü hedef input'un kendisidir.
                   loader = DataLoader(
                       TensorDataset(X_tr),
                       batch_size=bs,
                       shuffle=True
                   )
                   # Latent dimension seçenekleri denenir.
                   for ld in sp["latent_dims"]:
                       # Num layers seçenekleri denenir.
                       for nl in sp["num_layers"]:
                           # Dropout seçenekleri denenir.
                           for dr in sp["dropouts"]:
                               # Denenen Autoencoder parametreleri loglanır.
                               print(
                                   f"\n AE | lb={lb} | ld={ld} | nl={nl} | dr={dr} | lr={lr} | bs={bs}"
                               )
                               # Belirlenen parametrelerle Autoencoder modeli oluşturulur.
                               m = Autoencoder(
                                   lb,
                                   latent_dim=ld,
                                   num_layers=nl,
                                   dropout=dr
                               )
                               # Autoencoder eğitilir.
                               # ae=True olduğu için hedef input'un kendisi olur.
                               m, loss = _train_eval(
                                   m,
                                   loader,
                                   X_val,
                                   None,
                                   epochs,
                                   lr,
                                   device,
                                   ae=True
                               )
                               # Bu kombinasyonun loss değeri loglanır.
                               print(f" Model Loss = {loss:.6f}")
                               # Eğer bu Autoencoder şimdiye kadarki en iyi modelden daha iyiyse:
                               if loss < best_loss:
                                   # Yeni en iyi model bilgisi loglanır.
                                   print(" New BEST model found!")
                                   # En iyi loss güncellenir.
                                   best_loss = loss
                                   # En iyi model güncellenir.
                                   best_model = m
                                   # En iyi Autoencoder parametreleri kaydedilir.
                                   best_params = {
                                       "model_type": "autoencoder",
                                       "look_back": lb,
                                       "latent_dim": ld,
                                       "num_layers": nl,
                                       "dropout": dr,
                                       "lr": lr,
                                       "batch_size": bs
                                   }
   # Hyperparameter search bittiği loglanır.
   print("\n\n Hyperopt FINISHED!")
   # En iyi loss loglanır.
   print(f" Best Loss: {best_loss:.6f}")
   # En iyi parametreler loglanır.
   print(f" Best Params: {best_params}\n")
   # En iyi model, parametreleri ve loss değeri döndürülür.
   return best_model, best_params, best_loss