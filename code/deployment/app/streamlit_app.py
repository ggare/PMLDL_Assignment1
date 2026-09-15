import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("🔬 Diabetes Risk Predictor")
st.caption("Введите показатели пациента. Прогноз возвращает модель, развернутая в отдельном API-контейнере.")

col1, col2 = st.columns(2)
with col1:
    pregnancies = st.number_input("Беременностей", 0, 20, 1)
    glucose = st.number_input("Глюкоза (mg/dL)", 1.0, 300.0, 120.0)
    blood_pressure = st.number_input("Артериальное давление (mm Hg)", 1.0, 200.0, 70.0)
    skin_thickness = st.number_input("Толщина кожной складки (mm)", 1.0, 110.0, 25.0)
with col2:
    insulin = st.number_input("Инсулин (mu U/ml)", 1.0, 900.0, 80.0)
    bmi = st.number_input("Индекс массы тела (BMI)", 1.0, 80.0, 30.0)
    diabetes_pedigree = st.number_input("Функция наследственности (Pedigree)", 0.0, 3.0, 0.5)
    age = st.number_input("Возраст", 1, 120, 33)

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
        st.metric("Вероятность диабета", f"{result['diabetes_probability']:.1%}")
        if result["diabetes"]:
            st.error("Прогноз: ВЫСОКИЙ риск диабета")
        else:
            st.success("Прогноз: низкий риск диабета")
    except requests.RequestException as e:
        st.error(f"Не удалось обратиться к API модели: {e}")