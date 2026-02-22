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

preg = st.number_input("Pregnancies",value=1)
gluc = st.number_input("Glucose", value=120)
bp = st.number_input("Blood Pressure", value=70)
skin = st.number_input("Skin Thickness", value=20)
ins = st.number_input("Insulin", value=80)
bmi = st.number_input("BMI", value=25.0)
dpf = st.number_input("Pedigree Function", value=0.5)
age = st.number_input("Age", value=30)

if st.button("Predict Your Risk"):
    userdata =np.array([[preg,gluc,bp, skin,ins, bmi,dpf, age]])
    scaled = scaler.transform(userdata)
    probability = model.predict_proba(scaled)[0][1]
    st.subheader("Result")
    percent = round(probability * 100, 2)
    st.write(f"Your Probability of getting Diabetic: {percent}%")

    if probability >= 0.7:
        st.write("You may be highly Diabetic immediately consult a doctor.")
    elif probability >= 0.4:
        st.write("You are on the border line of Diabetic improve your lifestyle.")
    else:
        st.write("Your predicted score is low you are safe")

    st.subheader("Top Contributing Factors (Patient-Specific)")

    if hasattr(model, "coef_"):
        st.markdown("**Main contributing factors:**")

        features=["preg","gluc","bp","skin","ins","bmi","dpf","age"]
        coefs = model.coef_[0]
        values = scaled[0]
        contri=[]

        for i in range(len(features)):
            contri.append((feature[i],coefs[i]*values[i]))
        
        contri.sort()
        top3=contri[-3:]
        
        for i,j in top3:
            if j>0:
                st.write(f"{i}: {round(j, 3)} (increased risk)")
            else:
                st.write(f"{i}: {round(j, 3)} (reduced risk)")
st.divider()