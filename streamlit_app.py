import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="CrediShield - Credit Risk Engine",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("credit_model.pkl")

# -----------------------------
# Header Section
# -----------------------------
st.title("CrediShield — Enterprise Credit Risk Engine")

st.markdown("""
AI-driven credit default probability estimation system designed to support institutional lending decisions.
""")

st.divider()

# -----------------------------
# Input Section
# -----------------------------
st.header("Applicant Credit Profile")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Financial Information")
    amt_income = st.number_input("Annual Income", min_value=0.0, value=200000.0)
    amt_credit = st.number_input("Loan Amount", min_value=0.0, value=500000.0)
    amt_annuity = st.number_input("Loan Annuity", min_value=0.0, value=25000.0)

with col2:
    st.subheader("Credit Risk Indicators")
    ext_source_2 = st.number_input("External Score 2", min_value=0.0, max_value=1.0, value=0.5)
    ext_source_3 = st.number_input("External Score 3", min_value=0.0, max_value=1.0, value=0.5)

st.divider()

# -----------------------------
# Prediction Section
# -----------------------------
if st.button("Evaluate Applicant"):

    input_data = pd.DataFrame({
        "AMT_INCOME_TOTAL": [amt_income],
        "AMT_CREDIT": [amt_credit],
        "AMT_ANNUITY": [amt_annuity],
        "EXT_SOURCE_2": [ext_source_2],
        "EXT_SOURCE_3": [ext_source_3]
    })

    probability = model.predict_proba(input_data)[0][1]
    probability_percent = round(probability * 100, 2)

    st.header("Risk Assessment Result")
    st.markdown(f"### Estimated Default Probability: **{probability_percent}%**")

    if probability < 0.4:
        risk_level = "Low Risk"
        action = "Recommended Action: Approve Application"
    elif probability < 0.7:
        risk_level = "Moderate Risk"
        action = "Recommended Action: Manual Credit Review Required"
    else:
        risk_level = "High Risk"
        action = "Recommended Action: Decline Application"

    st.markdown(f"**Risk Classification:** {risk_level}")
    st.markdown(f"**{action}**")

    st.subheader("Risk Probability")

    progress_value = int(probability * 100)
    st.progress(progress_value)

    st.write(f"Risk Score: {progress_value}%")

    st.divider()

# -----------------------------
# Model Transparency Section
# -----------------------------
st.header("Model Information & Transparency")

st.markdown("""
- **Model Type:** XGBoost Classifier  
- **Training Dataset Size:** 246,000+ applicants  
- **Performance Metric:** ROC-AUC ≈ 0.72  
- **Imbalance Handling:** Applied during training  
- **Deployment:** Streamlit Web Application  

**Disclaimer:**  
This system provides probability-based risk estimates intended to assist lending professionals. Final credit decisions should incorporate institutional underwriting policies and human review.
""")



