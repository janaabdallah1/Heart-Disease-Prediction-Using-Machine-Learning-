import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Jana Mahmoud Abdalla - Heart Disease Prediction", page_icon="❤️", layout="wide")

st.markdown("""
<style>
.stApp {background-color: #111827; color: #f3f4f6;}
.block-container {padding-top: 2rem; max-width: 1100px;}
[data-testid="stSidebar"] {background-color: #0f172a;}
.title-card {padding: 1.4rem; border-radius: 18px; background: linear-gradient(135deg,#1f2937,#111827); border: 1px solid #374151; margin-bottom: 1rem;}
.result-high {padding: 1.2rem; border-radius: 14px; background-color: #7f1d1d; border: 1px solid #ef4444;}
.result-low {padding: 1.2rem; border-radius: 14px; background-color: #064e3b; border: 1px solid #10b981;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("best_heart_disease_model.pkl")

model = load_model()

st.sidebar.header("❤️ About the App")
st.sidebar.write("This app predicts heart disease risk using demographic, ECG, exercise, and clinical features.")
st.sidebar.markdown("**👩‍💻 Developed by**")
st.sidebar.write("Jana Mahmoud Abdalla")
st.sidebar.write("ID: 231001245")
st.sidebar.markdown("**📌 Model Note**")
st.sidebar.write("This model is an educational screening-support demonstration and does not replace medical diagnosis.")

st.markdown('<div class="title-card"><h1>❤️ Jana Mahmoud Abdalla Heart Disease Prediction App</h1><p>Enter patient information to estimate heart disease risk using the trained machine learning model.</p></div>', unsafe_allow_html=True)

st.subheader("🩺 Patient Information")
col1, col2 = st.columns(2)
with col1:
    sex = st.selectbox("Sex", ["M", "F"], format_func=lambda x: "Male" if x=="M" else "Female")
    age = st.slider("Age", 20, 90, 54)
    chest_pain = st.selectbox("Chest Pain Type", ["ASY", "ATA", "NAP", "TA"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 220, 130)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 650, 220)
    fasting_bs = st.radio("Fasting Blood Sugar > 120 mg/dL", [0,1], format_func=lambda x: "Yes" if x==1 else "No", horizontal=True)
with col2:
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "LVH", "ST"])
    max_hr = st.number_input("Maximum Heart Rate", 60, 220, 140)
    exercise_angina = st.radio("Exercise-Induced Angina", ["N","Y"], format_func=lambda x: "Yes" if x=="Y" else "No", horizontal=True)
    oldpeak = st.number_input("Oldpeak (ST Depression)", -3.0, 7.0, 1.0, step=0.1)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

input_df = pd.DataFrame([{
    "Age": age,
    "Sex": sex,
    "ChestPainType": chest_pain,
    "RestingBP": resting_bp,
    "Cholesterol": cholesterol,
    "FastingBS": fasting_bs,
    "RestingECG": resting_ecg,
    "MaxHR": max_hr,
    "ExerciseAngina": exercise_angina,
    "Oldpeak": oldpeak,
    "ST_Slope": st_slope
}])

input_df["age_group"] = pd.cut(input_df["Age"], bins=[0,39,49,59,100], labels=["Under 40","40-49","50-59","60+"])
input_df["bp_category"] = pd.cut(input_df["RestingBP"], bins=[0,119,129,139,300], labels=["Normal","Elevated","Stage 1","Stage 2"])
input_df["cholesterol_category"] = pd.cut(input_df["Cholesterol"], bins=[0,199,239,1000], labels=["Desirable","Borderline High","High"])
input_df["max_hr_percent"] = (input_df["MaxHR"] / (220 - input_df["Age"]) * 100).round(1)
input_df["exercise_capacity"] = pd.cut(input_df["max_hr_percent"], bins=[0,59.9,79.9,200], labels=["Low","Moderate","High"])

with st.expander("View automatically generated health categories"):
    st.write("Age Group:", str(input_df.loc[0,"age_group"]))
    st.write("Blood Pressure Category:", str(input_df.loc[0,"bp_category"]))
    st.write("Cholesterol Category:", str(input_df.loc[0,"cholesterol_category"]))
    st.write("Exercise Capacity:", str(input_df.loc[0,"exercise_capacity"]))
    st.write("Age-Predicted Max HR Reached:", f"{input_df.loc[0,'max_hr_percent']:.1f}%")

st.subheader("🔍 Prediction")
if st.button("Predict Heart Disease Risk", use_container_width=True):
    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0,1])
    st.metric("Predicted Heart Disease Probability", f"{probability*100:.1f}%")
    st.progress(min(max(probability,0.0),1.0))
    if prediction == 1:
        st.markdown(f'<div class="result-high"><h3>⚠️ Higher Heart Disease Risk Predicted</h3><p>The model predicts the positive class. Estimated probability: {probability*100:.1f}%.</p></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result-low"><h3>✅ Lower Heart Disease Risk Predicted</h3><p>The model predicts the negative class. Estimated probability of heart disease: {probability*100:.1f}%.</p></div>', unsafe_allow_html=True)
    st.subheader("Patient Input Summary")
    st.dataframe(input_df, use_container_width=True)
