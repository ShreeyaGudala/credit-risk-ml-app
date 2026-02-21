import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(
    page_title="CrediShield - Enterprise Risk Engine",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 CrediShield")
st.caption("Enterprise Credit Risk Assessment & Exposure Simulation")

st.divider()

# ----------------------------------
# LOAD MODEL
# ----------------------------------
@st.cache_resource
def load_model():
    return joblib.load("credit_model.pkl")

model = load_model()

# ----------------------------------
# INPUT SECTION
# ----------------------------------
st.subheader("Applicant Financial Profile")

MAX_LIMIT = 50_000_000.0  # ₹5 Crores

col1, col2 = st.columns(2)

with col1:
    amt_income = st.number_input(
        "Annual Income",
        min_value=0.0,
        max_value=MAX_LIMIT,
        value=0.0,
        step=10000.0
    )

    repayment_reliability = st.slider(
        "Repayment Reliability Score",
        0.0, 1.0, 0.5
    )

with col2:
    amt_credit = st.number_input(
        "Loan Amount Requested",
        min_value=0.0,
        max_value=MAX_LIMIT,
        value=0.0,
        step=10000.0
    )

    credit_history_strength = st.slider(
        "Credit History Strength",
        0.0, 1.0, 0.5
    )

amt_annuity = st.number_input(
    "Loan Annuity",
    min_value=0.0,
    max_value=MAX_LIMIT,
    value=0.0,
    step=5000.0
)

st.divider()

# ----------------------------------
# PREDICTION
# ----------------------------------
if st.button("Evaluate Risk"):

    if amt_income <= 0 or amt_credit <= 0 or amt_annuity <= 0:
        st.warning("All financial values must be greater than zero.")
        st.stop()

    input_data = pd.DataFrame({
        "AMT_INCOME_TOTAL": [amt_income],
        "AMT_CREDIT": [amt_credit],
        "AMT_ANNUITY": [amt_annuity],
        "EXT_SOURCE_2": [repayment_reliability],
        "EXT_SOURCE_3": [credit_history_strength]
    })

    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Default Probability")
    st.progress(float(probability))

    st.write(f"Estimated Probability of Default: **{probability:.2%}**")

    # ----------------------------------
    # PROFESSIONAL THRESHOLDS
    # ----------------------------------
    if probability < 0.05:
        st.success("Low Risk – Approval Recommended")
    elif probability < 0.12:
        st.warning("Moderate Risk – Further Credit Review Required")
    else:
        st.error("High Risk – Approval Not Recommended")

    st.divider()

    # ----------------------------------
    # FINANCIAL SIMULATION
    # ----------------------------------
    st.subheader("12-Month Exposure Simulation")

    months = np.arange(1, 13)
    interest_rate = 0.12
    monthly_interest = (amt_credit * interest_rate) / 12

    repaid_curve = months * monthly_interest
    default_curve = np.full(12, -amt_credit)
    expected_curve = (1 - probability) * repaid_curve + probability * default_curve

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(months, repaid_curve, linewidth=3, label="If Repaid")
    ax.plot(months, default_curve, linewidth=3, label="If Default")
    ax.plot(months, expected_curve, linestyle="--", linewidth=3, label="Expected Outcome")

    ax.set_xlabel("Months")
    ax.set_ylabel("Financial Impact")
    ax.set_title("Projected Loan Exposure")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.legend()

    st.pyplot(fig)
