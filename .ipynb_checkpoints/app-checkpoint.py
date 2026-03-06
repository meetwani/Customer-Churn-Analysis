import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📡",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

.main {
    background-color: #0d0d0d;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
}

.title-block {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-left: 5px solid #e94560;
    padding: 2rem 2rem 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 2rem;
}

.title-block h1 {
    color: #ffffff;
    font-size: 2.2rem;
    margin: 0;
    letter-spacing: -0.5px;
}

.title-block p {
    color: #a0aec0;
    margin: 0.5rem 0 0 0;
    font-size: 0.95rem;
    font-family: 'Space Mono', monospace;
}

.section-label {
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    color: #e94560;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.5rem;
    margin-top: 1.5rem;
}

.result-churn {
    background: linear-gradient(135deg, #3d0000, #7b0000);
    border: 2px solid #e94560;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
}

.result-nochurn {
    background: linear-gradient(135deg, #003d1a, #006630);
    border: 2px solid #00d68f;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
}

.result-emoji {
    font-size: 3rem;
    display: block;
    margin-bottom: 0.5rem;
}

.result-title {
    font-size: 1.8rem;
    font-weight: 800;
    color: white;
    margin: 0;
}

.result-subtitle {
    color: #cbd5e0;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    margin-top: 0.5rem;
}

.prob-bar-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: #a0aec0;
}

.stButton > button {
    background: linear-gradient(135deg, #e94560, #c62a47);
    color: white;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.1rem;
    border: none;
    border-radius: 10px;
    padding: 0.75rem 2rem;
    width: 100%;
    letter-spacing: 0.5px;
    transition: all 0.3s;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #ff6b6b, #e94560);
    transform: translateY(-1px);
}

div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stSlider"] label {
    color: #cbd5e0 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.82rem !important;
}

.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background-color: #1a1a2e !important;
    color: white !important;
    border: 1px solid #2d3748 !important;
    border-radius: 8px !important;
}

.info-card {
    background: #1a1a2e;
    border: 1px solid #2d3748;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: #a0aec0;
}

.info-card strong {
    color: #e94560;
}
</style>
""", unsafe_allow_html=True)

# ── Load Model & Encoders ─────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("customer churn model.pkl", "rb") as f:
        model_data = pickle.load(f)
    with open("encoders.pikle", "rb") as f:
        encoders = pickle.load(f)
    return model_data["model"], model_data["features_names"], encoders

try:
    model, feature_names, encoders = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"⚠️ Model load error: {e}")

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="title-block">
    <h1>📡 Telco Churn Predictor</h1>
    <p>// Enter customer details below to predict churn probability</p>
</div>
""", unsafe_allow_html=True)

if model_loaded:

    # ── Form ──────────────────────────────────────────────────
    st.markdown('<div class="section-label">👤 Customer Demographics</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        partner = st.selectbox("Partner", ["Yes", "No"])
    with col2:
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        dependents = st.selectbox("Dependents", ["No", "Yes"])

    st.markdown('<div class="section-label">📋 Account Info</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    with col4:
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=0.5)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, round(tenure * 65.0, 2), step=1.0)
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])

    st.markdown('<div class="section-label">📶 Services</div>', unsafe_allow_html=True)
    col5, col6, col7 = st.columns(3)
    with col5:
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multi = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    with col6:
        online_sec = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_bkp = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    with col7:
        tech = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    st.markdown("---")

    # ── Predict Button ────────────────────────────────────────
    if st.button("🔍 Predict Churn"):
        input_data = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multi,
            "InternetService": internet,
            "OnlineSecurity": online_sec,
            "OnlineBackup": online_bkp,
            "DeviceProtection": device,
            "TechSupport": tech,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        input_df = pd.DataFrame([input_data])

        # Encode categorical columns
        for col, encoder in encoders.items():
            if col in input_df.columns:
                input_df[col] = encoder.transform(input_df[col])

        # Reorder columns to match training
        input_df = input_df[feature_names]

        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0]

        churn_prob = round(prob[1] * 100, 1)
        stay_prob = round(prob[0] * 100, 1)

        if prediction == 1:
            st.markdown(f"""
            <div class="result-churn">
                <span class="result-emoji">⚠️</span>
                <p class="result-title">HIGH CHURN RISK</p>
                <p class="result-subtitle">This customer is likely to leave — consider a retention offer</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-nochurn">
                <span class="result-emoji">✅</span>
                <p class="result-title">LOW CHURN RISK</p>
                <p class="result-subtitle">This customer is likely to stay — no immediate action needed</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-label">📊 Prediction Probability</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("🔴 Churn Probability", f"{churn_prob}%")
            st.progress(prob[1])
        with col_b:
            st.metric("🟢 Stay Probability", f"{stay_prob}%")
            st.progress(prob[0])

        # Business insight
        st.markdown('<div class="section-label">💡 Business Insight</div>', unsafe_allow_html=True)
        if contract == "Month-to-month" and churn_prob > 50:
            st.info("📌 Month-to-month contract + high churn risk → Offer a discounted annual plan")
        elif tenure < 6 and churn_prob > 40:
            st.info("📌 New customer (low tenure) + churn risk → Onboarding support may help retain")
        elif monthly_charges > 80 and churn_prob > 50:
            st.info("📌 High monthly charges → Consider offering a loyalty discount")
        else:
            st.success("📌 Customer appears stable. Continue regular engagement.")
