import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")
st.title("🚢 Titanic Survival Predictor")

# Check if model exists
if not os.path.exists("titanic_model.pkl"):
    st.error("❌ Model file not found. Please run train_model.py.")
    st.stop()

# Load model
model = joblib.load("titanic_model.pkl")

# Collect user inputs
st.header("🧾 Passenger Information")

pclass = st.selectbox("Passenger Class", [1, 2, 3], index=0)
sex = st.radio("Sex", ["male", "female"], index=0)
age = st.slider("Age", 0, 80, 30)
sibsp = st.slider("Number of Siblings/Spouses Aboard", 0, 8, 0)
fare = st.slider("Ticket Fare ($)", 0.0, 500.0, 50.0)

# Encode input
sex_encoded = 0 if sex == "male" else 1
features = np.array([[pclass, sex_encoded, age, sibsp, fare]])

# Make prediction
prediction = model.predict(features)[0]
prediction_proba = model.predict_proba(features)[0]

# Show result
st.subheader("🧠 Prediction Result")
if prediction == 1:
    st.success("✅ The passenger **would survive**.")
else:
    st.error("❌ The passenger **would not survive**.")

# Show probabilities
st.markdown("### 🔍 Prediction Probabilities")
st.write(f"**Survive:** {prediction_proba[1]*100:.2f}%")
st.write(f"**Not Survive:** {prediction_proba[0]*100:.2f}%")
