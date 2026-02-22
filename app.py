import pandas as pd
import streamlit as st
import numpy as np
import joblib


st.set_page_config(page_title="Diabetes Risk Prediction",layout="centered")
st.title("Diabetes Risk Prediction System")
st.write("Predict diabetes risk by just answering few simple questions")
st.divider()
st.subheader("Enter Patient Details")

model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

preg = st.number_input("Pregnancies", 0, 20, 1)
gluc = st.number_input("Glucose", 50, 250, 120)
bp = st.number_input("Blood Pressure", 40, 200, 70)
skin = st.number_input("Skin Thickness", 5, 100, 20)
ins = st.number_input("Insulin", 15, 900, 80)
bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
dpf = st.number_input("Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 18, 100, 30)

if st.button("Predict Risk"):
    user_data =np.array([[preg, gluc, bp, skin, ins, bmi, dpf, age]])
    try:
        user_scaled = scaler.transform(user_data)
        probability = model.predict_proba(user_scaled)[0][1]
    except Exception:
        st.error("Prediction failed. Please try again.")
        st.stop()
    st.subheader("Result")
    percent = round(probability * 100, 2)
    st.write(f"Probability: {percent}%")

    if probability >= 0.7:
        st.error("High risk detected. Please consult a doctor.")
    elif probability >= 0.4:
        st.warning("Moderate risk. Consider lifestyle improvements.")
    else:
        st.success("Low risk.")

    st.divider()
    st.subheader("Top Contributing Factors (Patient-Specific)")

    if hasattr(model, "coef_"):
        st.markdown("**Main contributing factors:**")

        coefs = model.coef_[0]
        values = user_scaled[0]

        impact = coefs * values

        pairs = list(zip(
            ["Preg","Glucose","BP","Skin","Insulin","BMI","DPF","Age"],
            impact
        ))

        pairs.sort(key=lambda x: abs(x[1]), reverse=True)

        for feature, value in pairs[:3]:
            sign = "increased" if value > 0 else "reduced"
            st.write(f"{feature}: {round(value, 3)} ({sign} risk)")
st.divider()