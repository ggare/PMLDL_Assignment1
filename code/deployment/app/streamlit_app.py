import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("Diabetes Risk Predictor")
st.caption("Enter the patient’s metrics. The forecast is returned by a model deployed in a separate API container.")

col1, col2 = st.columns(2)
with col1:
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose (mg/dL)", 1.0, 300.0, 120.0)
    blood_pressure = st.number_input("Blood pressure (mm Hg)", 1.0, 200.0, 70.0)
    skin_thickness = st.number_input("The thickness of the skin fold (mm)", 1.0, 110.0, 25.0)
with col2:
    insulin = st.number_input("Insulin (mu U/ml)", 1.0, 900.0, 80.0)
    bmi = st.number_input("Body mass index (BMI)", 1.0, 80.0, 30.0)
    diabetes_pedigree = st.number_input("The function of heredity (Pedigree)", 0.0, 3.0, 0.5)
    age = st.number_input("Age", 1, 120, 33)

if st.button("Predict", type="primary"):
    payload = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree,
        "Age": age,
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        st.metric("Diabetes probability", f"{result['diabetes_probability']:.1%}")
        if result["diabetes"]:
            st.error("Forecast: HIGH risk of diabetes")
        else:
            st.success("Forecast: low risk of diabetes")
    except requests.RequestException as e:
        st.error(f"Failed to access the model’s API.: {e}")