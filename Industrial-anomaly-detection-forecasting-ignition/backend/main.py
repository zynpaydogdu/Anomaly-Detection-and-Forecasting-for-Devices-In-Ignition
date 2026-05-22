from fastapi import FastAPI
import os
# Ayırdığımız yönlendiricileri (router) içe aktarıyoruz
from router_train import router as train_router
from router_infer import router as infer_router
app = FastAPI()
# Klasörlerin var olduğundan emin olalım
os.makedirs("models", exist_ok=True)
os.makedirs("datasets", exist_ok=True)
# İki ayrı dosyayı ana uygulamamıza bağlıyoruz
app.include_router(train_router)
app.include_router(infer_router)
@app.get("/")
def root():
   return {"status": "running", "message": "Anomaly + Forecast API aktif"}
@app.get("/health")
def health():
   return {"status": "ok"}