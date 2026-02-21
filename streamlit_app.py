import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="CrediShield - Credit Risk Engine",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 CrediShield - Enterprise Credit Risk Engine")
st.markdown("AI-powered credit risk evaluation & financial impact simulation.")

st.divider()

# -----------------------------------
# LOAD MODEL
# -----------------------------------
@st.cache_resource
def load_model():
    return joblib.load("credit_model.pkl")

model = load_model()

# -----------------------------------
# USER INPUT SECTION
# -----------------------------------
st.subheader("Applicant Financial Details")

amt_income = st.number_input(
    "Annual Income",
    min_value=0.0,
    max_value=5_00_00_000.0,
    value=None,
    placeholder="Enter annual income"
)

amt_credit = st.number_input(
    "Loan Amount Requested",
    min_value=0.0,
    max_value=5_00_00_000.0,
    value=None,
    placeholder="Enter requested loan amount"
)

amt_annuity = st.number_input(
    "Loan Annuity",
    min_value=0.0,
    max_value=5_00_00_000.0,
    value=None,
    placeholder="Enter annuity amount"
)

repayment_reliability = st.number_input(
    "Repayment Reliability Score",
    min_value=0.0,
    max_value=1.0,
    value=None,
    placeholder="0.0 to 1.0",
    help="Higher score indicates stronger past repayment behavior."
)

credit_history_strength = st.number_input(
    "Credit History Strength",
    min_value=0.0,
    max_value=1.0,
    value=None,
    placeholder="0.0 to 1.0",
    help="Represents overall strength and depth of credit history."
)

st.divider()

# -----------------------------------
# PREDICTION BUTTON
# -----------------------------------
if st.button("Predict Risk"):

    # Validate inputs
    if None in [
        amt_income,
        amt_credit,
        amt_annuity,
        repayment_reliability,
        credit_history_strength
    ]:
        st.warning("Please fill in all required fields.")
    else:

        # Create dataframe EXACTLY matching training features
        input_data = pd.DataFrame({
            "AMT_INCOME_TOTAL": [amt_income],
            "AMT_CREDIT": [amt_credit],
            "AMT_ANNUITY": [amt_annuity],
            "EXT_SOURCE_2": [repayment_reliability],
            "EXT_SOURCE_3": [credit_history_strength]
        })

        probability = model.predict_proba(input_data)[0][1]

        st.subheader(f"Predicted Default Probability: {probability:.2%}")

        # -----------------------------------
        # RISK CLASSIFICATION
        # -----------------------------------
        if probability < 0.30:
            st.success("Low Risk Applicant – Loan Approval Recommended")
        elif probability < 0.60:
            st.warning("Moderate Risk – Manual Credit Review Required")
        else:
            st.error("High Risk – Loan Rejection Recommended")

        st.divider()

        # -----------------------------------
        # FINANCIAL IMPACT SIMULATION
        # -----------------------------------
        st.subheader("12-Month Financial Exposure Projection")

        months = np.arange(1, 13)
        interest_rate = 0.12
        monthly_interest = (amt_credit * interest_rate) / 12

        repaid_curve = months * monthly_interest
        default_curve = np.full(12, -amt_credit)
        expected_curve = (1 - probability) * repaid_curve + probability * default_curve

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(repaid_curve, months, linewidth=3, label="If Repaid")
        ax.plot(default_curve, months, linewidth=3, label="If Default")
        ax.plot(expected_curve, months, linestyle="--", linewidth=3, label="Expected Outcome")

        ax.set_ylabel("Months")
        ax.set_xlabel("Amount")
        ax.set_title("Loan Exposure Simulation")

        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.legend()

        st.pyplot(fig)
