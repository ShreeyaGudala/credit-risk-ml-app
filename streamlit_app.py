import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="CrediShield - Credit Risk Engine",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 CrediShield - Enterprise Credit Risk Engine")
st.markdown("AI-powered credit risk evaluation & financial impact simulation.")

st.divider()

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("credit_model.pkl")

model = load_model()

# -------------------------------
# USER INPUT SECTION
# -------------------------------
st.subheader("Applicant Financial Details")

amt_income = st.number_input("Annual Income (₹)", min_value=0.0, value=300000.0)
amt_credit = st.number_input("Loan Amount Requested (₹)", min_value=0.0, value=500000.0)
amt_annuity = st.number_input("Loan Annuity (₹)", min_value=0.0, value=25000.0)
ext_source_2 = st.number_input("External Risk Score 1", min_value=0.0, max_value=1.0, value=0.5)
ext_source_3 = st.number_input("External Risk Score 2", min_value=0.0, max_value=1.0, value=0.5)

st.divider()

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Evaluate Credit Risk"):

    input_data = pd.DataFrame({
        "AMT_INCOME_TOTAL": [amt_income],
        "AMT_CREDIT": [amt_credit],
        "AMT_ANNUITY": [amt_annuity],
        "EXT_SOURCE_2": [ext_source_2],
        "EXT_SOURCE_3": [ext_source_3]
    })

    probability = model.predict_proba(input_data)[0][1]

    # -------------------------------
    # RISK CLASSIFICATION
    # -------------------------------
    if probability < 0.3:
        risk_label = "Low Risk"
        recommendation = "Loan can be Approved"
        st.success(f"Risk Classification: {risk_label}")
    elif probability < 0.6:
        risk_label = "Moderate Risk"
        recommendation = "Manual Credit Review Required"
        st.warning(f"Risk Classification: {risk_label}")
    else:
        risk_label = "High Risk"
        recommendation = "Loan Rejection Recommended"
        st.error(f"Risk Classification: {risk_label}")

    st.subheader(f"Recommended Action: {recommendation}")

    st.divider()

    # -------------------------------
    # FINANCIAL IMPACT SIMULATION
    # -------------------------------
    st.subheader("Financial Impact Analysis")

    interest_rate = 0.12  # 12% assumed

    profit_if_repaid = amt_credit * interest_rate
    loss_if_default = amt_credit

    expected_value = (1 - probability) * profit_if_repaid - probability * loss_if_default

    scenarios = ["Repaid", "Default", "Expected Outcome"]
    values = [profit_if_repaid, -loss_if_default, expected_value]

    fig, ax = plt.subplots()

    colors = ["green", "red", "blue"]
    ax.bar(scenarios, values, color=colors)

    ax.set_ylabel("Amount (₹)")
    ax.set_title("Loan Financial Outcome Projection")

    st.pyplot(fig)

    st.caption("Expected Outcome represents risk-adjusted financial impact.")

    st.divider()

    # -------------------------------
    # RISK SCORE DISPLAY
    # -------------------------------
    st.subheader("Risk Score")
    risk_percent = int(probability * 100)
    st.metric("Default Probability", f"{risk_percent}%")
