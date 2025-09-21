import streamlit as st
import numpy as np
import joblib
import os

st.title("❤ Personal Health Recommendation System")
st.write("This is a demo project. Not medical advice.")

# Path to model
model_path = r"c:\Users\jayasurya\Downloads\heart_model.joblib"

# Check if file exists
if not os.path.exists(model_path):
    st.error(f"❌ Model file not found at: {model_path}")
    st.stop()

# Load model
model = joblib.load(model_path)

# Collect user input
age = st.number_input("Age", 18, 120, 30)
sex = st.selectbox("Sex", ("Female", "Male"))
cp = st.selectbox("Chest pain type", [0, 1, 2, 3])
trestbps = st.number_input("Resting blood pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 400, 200)
fbs = st.selectbox("Fasting blood sugar > 120mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG results", [0, 1, 2])
thalach = st.number_input("Max heart rate achieved", 60, 220, 150)
exang = st.selectbox("Exercise induced angina", [0, 1])
oldpeak = st.number_input("ST depression (oldpeak)", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope", [0, 1, 2])
ca = st.selectbox("Number of major vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thal", [1, 2, 3])

# Convert inputs to model format
sex_val = 1 if sex == "Male" else 0
features = np.array([[age, sex_val, cp, trestbps, chol, fbs,
                      restecg, thalach, exang, oldpeak, slope, ca, thal]])

# Predict
if st.button("Check Result"):
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    st.subheader("Result")
    if prediction == 1:
        st.error(f"⚠ Higher chance of heart disease (probability: {prob:.2f})")

        st.subheader("🩺 Recommendations")
        st.write("- Consult a cardiologist or healthcare professional soon")
        st.write("- Monitor blood pressure, sugar, and cholesterol regularly")
        st.write("- Adopt a heart-healthy diet (low salt, low sugar, more veggies & fruits)")
        st.write("- Exercise moderately (walking, yoga, light cardio)")
        st.write("- Quit smoking & reduce alcohol intake")
        st.write("- Reduce stress (meditation, deep breathing, enough sleep)")

    else:
        st.success(f"✅ Lower chance of heart disease (probability: {prob:.2f})")

        st.subheader("💡 Recommendations")
        st.write("- Maintain your healthy lifestyle")
        st.write("- Stay physically active (at least 30 mins/day of exercise)")
        st.write("- Eat a balanced diet rich in fruits, vegetables, and whole grains")
        st.write("- Get regular health checkups for early detection")
        st.write("- Keep stress under control and sleep well")
        st.write("- Stay hydrated and avoid junk food")