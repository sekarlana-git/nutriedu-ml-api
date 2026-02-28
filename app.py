from fastapi import FastAPI
import numpy as np
import joblib
import tensorflow as tf

app = FastAPI()

# Load model dan scaler
model = tf.keras.models.load_model("model.h5")
scaler = joblib.load("scaler.pkl")

# Mapping label sesuai training
labels = ["normal", "severely stunting", "stunting", "tinggi"]

@app.get("/")
def root():
    return {"message": "NutriEdu ML API is running 🚀"}

@app.post("/predict")
def predict(data: dict):
    umur = data["umur"]
    tinggi = data["tinggi"]
    jenis_kelamin = data["jenis_kelamin"]

    # Encode gender
    if jenis_kelamin == "laki-laki":
        gender_encoded = [1, 0]
    else:
        gender_encoded = [0, 1]

    # Susun input sesuai training
    X = np.array([[umur, tinggi] + gender_encoded])
    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)
    predicted_class = np.argmax(prediction)

    return {
        "status_gizi": labels[predicted_class]
    }