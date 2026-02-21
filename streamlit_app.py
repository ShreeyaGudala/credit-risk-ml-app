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
st.subheader("Financial Projection (12-Month Simulation)")

months = np.arange(1, 13)
interest_rate = 0.12
monthly_interest = (amt_credit * interest_rate) / 12

# Repaid scenario (cumulative growth)
repaid_curve = months * monthly_interest

# Default scenario (loss occurs immediately)
default_curve = np.full(12, -amt_credit)

# Expected value curve
expected_curve = (1 - probability) * repaid_curve + probability * default_curve

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(months, repaid_curve, label="If Repaid", linewidth=3)
ax.plot(months, default_curve, label="If Default", linewidth=3)
ax.plot(months, expected_curve, linestyle="--", linewidth=3, label="Expected Outcome")

ax.set_xlabel("Months")
ax.set_ylabel("Amount (₹)")
ax.set_title("Loan Cashflow Projection")
ax.legend()

st.pyplot(fig)
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
    st.subheader("Financial Projection (12-Month Simulation)")

months = np.arange(1, 13)
interest_rate = 0.12
monthly_interest = (amt_credit * interest_rate) / 12

# Repaid scenario
repaid_curve = months * monthly_interest

# Default scenario (flat loss)
default_curve = np.full(12, -amt_credit)

# Expected scenario
expected_curve = (1 - probability) * repaid_curve + probability * default_curve

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(repaid_curve, months, linewidth=3, label="If Repaid")
ax.plot(default_curve, months, linewidth=3, label="If Default")
ax.plot(expected_curve, months, linestyle="--", linewidth=3, label="Expected Outcome")

ax.set_ylabel("Months")
ax.set_xlabel("Amount (₹)")
ax.set_title("Loan Exposure Projection")

ax.axvline(0, color="gray", linestyle="--", linewidth=1)  # zero reference line

ax.legend()

st.pyplot(fig)


