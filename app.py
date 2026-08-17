from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Risk | Jana Mahmoud Abdalla",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------
# CUSTOM STYLE — CLEAN MEDICAL DASHBOARD
# ------------------------------------------------------------
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #f8fbff 0%, #eef5fb 100%);
            color: #183153;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 1.7rem;
            padding-bottom: 2.5rem;
        }

        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #dce7f0;
        }

        [data-testid="stSidebar"] * {
            color: #183153;
        }

        .hero {
            background: linear-gradient(120deg, #ffffff 0%, #eaf5ff 100%);
            border: 1px solid #d4e6f5;
            border-radius: 24px;
            padding: 2rem 2.1rem;
            margin-bottom: 1.4rem;
            box-shadow: 0 10px 35px rgba(28, 73, 112, 0.08);
        }

        .hero-badge {
            display: inline-block;
            background: #dff2ff;
            color: #176b9c;
            border-radius: 999px;
            padding: 0.35rem 0.8rem;
            font-size: 0.85rem;
            font-weight: 700;
            margin-bottom: 0.8rem;
        }

        .hero h1 {
            margin: 0;
            font-size: 2.35rem;
            color: #123a5a;
            line-height: 1.15;
        }

        .hero p {
            color: #56758d;
            font-size: 1.03rem;
            margin: 0.7rem 0 0 0;
        }

        .section-card {
            background: rgba(255, 255, 255, 0.94);
            border: 1px solid #dce8f1;
            border-radius: 20px;
            padding: 1.15rem 1.25rem 0.4rem 1.25rem;
            box-shadow: 0 7px 24px rgba(31, 74, 108, 0.05);
            margin-bottom: 1rem;
        }

        .mini-card {
            background: #ffffff;
            border: 1px solid #dce8f1;
            border-radius: 16px;
            padding: 1rem 1.1rem;
            min-height: 112px;
            box-shadow: 0 5px 18px rgba(31, 74, 108, 0.05);
        }

        .mini-label {
            color: #6b879b;
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .mini-value {
            color: #153e5c;
            font-size: 1.28rem;
            font-weight: 800;
            margin-top: 0.25rem;
        }

        .risk-low,
        .risk-high {
            border-radius: 20px;
            padding: 1.4rem 1.5rem;
            margin-top: 0.7rem;
        }

        .risk-low {
            background: #eafaf2;
            border: 1px solid #a8e3c3;
            color: #146b43;
        }

        .risk-high {
            background: #fff1f2;
            border: 1px solid #f5b6be;
            color: #9b1c31;
        }

        .risk-low h3,
        .risk-high h3 {
            margin: 0 0 0.35rem 0;
        }

        .footer-note {
            color: #7890a2;
            font-size: 0.82rem;
            text-align: center;
            margin-top: 2rem;
        }

        div.stButton > button {
            border-radius: 14px;
            min-height: 3.2rem;
            font-size: 1rem;
            font-weight: 750;
            border: none;
            background: linear-gradient(90deg, #176b9c, #1f89bd);
            color: white;
            box-shadow: 0 7px 18px rgba(23, 107, 156, 0.18);
        }

        div.stButton > button:hover {
            border: none;
            color: white;
            transform: translateY(-1px);
        }

        div[data-testid="stMetric"] {
            background: white;
            border: 1px solid #dce8f1;
            padding: 0.85rem 1rem;
            border-radius: 16px;
        }

        .stProgress > div > div > div > div {
            border-radius: 999px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# MODEL LOADING
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    base_dir = Path(__file__).resolve().parent
    candidates = [
        base_dir / "best_heart_disease_model.pkl",
        base_dir.parent / "best_heart_disease_model.pkl",
    ]

    for model_path in candidates:
        if model_path.exists():
            return joblib.load(model_path)

    raise FileNotFoundError(
        "best_heart_disease_model.pkl was not found. "
        "Place it in the same folder as app.py or one folder above it."
    )


try:
    model = load_model()
except Exception as exc:
    st.error(f"Unable to load the trained model: {exc}")
    st.stop()


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🫀 Heart Risk AI")
    st.caption("Machine Learning Screening Demo")
    st.divider()

    st.markdown("### 👩‍💻 Student")
    st.write("**Jana Mahmoud Abdalla**")
    st.write("ID: **231001245**")

    st.markdown("### 🎓 Course")
    st.write("CBIO313 — Data Mining and Machine Learning")

    st.markdown("### 🧠 Model")
    st.write("Tuned Random Forest classifier")

    st.divider()
    st.info(
        "This application is intended for educational demonstration only and is not a medical diagnostic tool."
    )


# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">CARDIOVASCULAR RISK SCREENING</div>
        <h1>Heart Disease Prediction Dashboard</h1>
        <p>
            Enter the patient's clinical and exercise-related measurements to estimate
            the probability of heart disease using the trained machine learning model.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# INPUT FORM
# ------------------------------------------------------------
st.markdown("### 🩺 Patient Assessment")
st.caption("Complete the fields below, then select **Analyze Heart Disease Risk**.")

with st.form("heart_disease_form", clear_on_submit=False):
    left, middle, right = st.columns(3, gap="large")

    with left:
        st.markdown("#### Patient Profile")
        age = st.slider("Age", min_value=20, max_value=90, value=54)
        sex = st.selectbox(
            "Sex",
            options=["M", "F"],
            format_func=lambda x: "Male" if x == "M" else "Female",
        )
        chest_pain = st.selectbox(
            "Chest Pain Type",
            options=["ASY", "ATA", "NAP", "TA"],
            format_func=lambda x: {
                "ASY": "Asymptomatic",
                "ATA": "Atypical Angina",
                "NAP": "Non-Anginal Pain",
                "TA": "Typical Angina",
            }[x],
        )
        fasting_bs = st.radio(
            "Fasting Blood Sugar > 120 mg/dL",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            horizontal=True,
        )

    with middle:
        st.markdown("#### Clinical Measurements")
        resting_bp = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            min_value=80,
            max_value=220,
            value=130,
            step=1,
        )
        cholesterol = st.number_input(
            "Cholesterol (mg/dL)",
            min_value=100,
            max_value=650,
            value=220,
            step=1,
        )
        resting_ecg = st.selectbox(
            "Resting ECG",
            options=["Normal", "LVH", "ST"],
            format_func=lambda x: {
                "Normal": "Normal",
                "LVH": "Left Ventricular Hypertrophy",
                "ST": "ST-T Wave Abnormality",
            }[x],
        )
        max_hr = st.number_input(
            "Maximum Heart Rate",
            min_value=60,
            max_value=220,
            value=140,
            step=1,
        )

    with right:
        st.markdown("#### Exercise / ECG Response")
        exercise_angina = st.radio(
            "Exercise-Induced Angina",
            options=["N", "Y"],
            format_func=lambda x: "Yes" if x == "Y" else "No",
            horizontal=True,
        )
        oldpeak = st.number_input(
            "Oldpeak (ST Depression)",
            min_value=-3.0,
            max_value=7.0,
            value=1.0,
            step=0.1,
        )
        st_slope = st.selectbox(
            "ST Slope",
            options=["Up", "Flat", "Down"],
            format_func=lambda x: {
                "Up": "Upsloping",
                "Flat": "Flat",
                "Down": "Downsloping",
            }[x],
        )

        st.write("")
        st.write("")
        submitted = st.form_submit_button(
            "🔍 Analyze Heart Disease Risk",
            use_container_width=True,
        )


# ------------------------------------------------------------
# BUILD INPUT DATAFRAME + FEATURE ENGINEERING
# ------------------------------------------------------------
input_df = pd.DataFrame(
    [
        {
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
            "ST_Slope": st_slope,
        }
    ]
)

input_df["age_group"] = pd.cut(
    input_df["Age"],
    bins=[0, 39, 49, 59, 100],
    labels=["Under 40", "40-49", "50-59", "60+"],
)

input_df["bp_category"] = pd.cut(
    input_df["RestingBP"],
    bins=[0, 119, 129, 139, 300],
    labels=["Normal", "Elevated", "Stage 1", "Stage 2"],
)

input_df["cholesterol_category"] = pd.cut(
    input_df["Cholesterol"],
    bins=[0, 199, 239, 1000],
    labels=["Desirable", "Borderline High", "High"],
)

input_df["max_hr_percent"] = (
    input_df["MaxHR"] / (220 - input_df["Age"]) * 100
).round(1)

input_df["exercise_capacity"] = pd.cut(
    input_df["max_hr_percent"],
    bins=[0, 59.9, 79.9, 200],
    labels=["Low", "Moderate", "High"],
)


# ------------------------------------------------------------
# LIVE DERIVED INFORMATION
# ------------------------------------------------------------
st.markdown("### 📋 Automatically Derived Categories")
card1, card2, card3, card4 = st.columns(4)

cards = [
    (card1, "Age Group", str(input_df.loc[0, "age_group"])),
    (card2, "Blood Pressure", str(input_df.loc[0, "bp_category"])),
    (card3, "Cholesterol", str(input_df.loc[0, "cholesterol_category"])),
    (card4, "Exercise Capacity", str(input_df.loc[0, "exercise_capacity"])),
]

for column, label, value in cards:
    with column:
        st.markdown(
            f"""
            <div class="mini-card">
                <div class="mini-label">{label}</div>
                <div class="mini-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.caption(
    f"Age-predicted maximum heart rate reached: "
    f"{input_df.loc[0, 'max_hr_percent']:.1f}%"
)


# ------------------------------------------------------------
# PREDICTION OUTPUT
# ------------------------------------------------------------
if submitted:
    try:
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0, 1])
    except Exception as exc:
        st.error(f"Prediction could not be generated: {exc}")
        st.stop()

    st.markdown("---")
    st.markdown("### 📊 Prediction Result")

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric("Heart Disease Probability", f"{probability * 100:.1f}%")

    with metric2:
        st.metric(
            "Predicted Class",
            "Heart Disease" if prediction == 1 else "No Heart Disease",
        )

    with metric3:
        st.metric(
            "Model Decision",
            "Positive Class" if prediction == 1 else "Negative Class",
        )

    st.write("")
    st.progress(min(max(probability, 0.0), 1.0))

    if prediction == 1:
        st.markdown(
            f"""
            <div class="risk-high">
                <h3>⚠️ Higher Heart Disease Risk Predicted</h3>
                <p>
                    The machine learning model classified this case as the positive
                    heart-disease class with an estimated probability of
                    <strong>{probability * 100:.1f}%</strong>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="risk-low">
                <h3>✅ Lower Heart Disease Risk Predicted</h3>
                <p>
                    The machine learning model classified this case as the negative
                    heart-disease class. The estimated probability of heart disease is
                    <strong>{probability * 100:.1f}%</strong>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("View complete model input"):
        st.dataframe(input_df, use_container_width=True, hide_index=True)


# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown(
    """
    <div class="footer-note">
        Jana Mahmoud Abdalla • 231001245 • CBIO313 Data Mining and Machine Learning<br>
        Educational machine learning demonstration — not a substitute for clinical diagnosis.
    </div>
    """,
    unsafe_allow_html=True,
)
