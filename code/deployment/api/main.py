from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Diabetes Prediction API", version="1.0.0")


class Patient(BaseModel):
    Pregnancies: int = Field(..., ge=0, le=20)
    Glucose: float = Field(..., ge=1, le=300)
    BloodPressure: float = Field(..., ge=1, le=200)
    SkinThickness: float = Field(..., ge=1, le=110)
    Insulin: float = Field(..., ge=1, le=900)
    BMI: float = Field(..., ge=1, le=80)
    DiabetesPedigreeFunction: float = Field(..., ge=0, le=3)
    Age: int = Field(..., ge=1, le=120)


@app.get("/")
def root():
    return {"service": "Diabetes Prediction API",
            "endpoints": ["GET /health", "POST /predict"]}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(patient: Patient):
    features = pd.DataFrame([patient.model_dump()])
    probability = float(model.predict_proba(features)[0, 1])
    return {
        "diabetes": bool(probability >= 0.5),
        "diabetes_probability": round(probability, 4),
    }