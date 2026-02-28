from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import tensorflow as tf

app = FastAPI()

# Load model dan scaler
model = tf.keras.models.load_model("model.h5")
scaler = joblib.load("scaler.pkl")

labels = ["normal", "severely stunting", "stunting", "tinggi"]

# 🔹 Schema validasi request
class PredictRequest(BaseModel):
    umur: int
    tinggi: float
    jenis_kelamin: str

@app.get("/")
def root():
    return {"message": "NutriEdu ML API is running 🚀"}

@app.post("/predict")
def predict(data: PredictRequest):
    try:
        umur = data.umur
        tinggi = data.tinggi
        jenis_kelamin = data.jenis_kelamin.lower()

        if jenis_kelamin == "laki-laki":
            gender_encoded = [1, 0]
        elif jenis_kelamin == "perempuan":
            gender_encoded = [0, 1]
        else:
            return {"error": "jenis_kelamin harus 'laki-laki' atau 'perempuan'"}

        X = np.array([[umur, tinggi] + gender_encoded])
        X_scaled = scaler.transform(X)

        prediction = model.predict(X_scaled)
        predicted_class = int(np.argmax(prediction))

        return {
            "status_gizi": labels[predicted_class]
        }

    except Exception as e:
        return {"error": str(e)}