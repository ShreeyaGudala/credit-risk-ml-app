import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("credit_model.pkl")

st.title("🏦 Credit Risk Prediction App")

st.write("Enter applicant details below:")

# --- User Inputs ---
amt_income = st.number_input("Income", value=150000.0)
amt_credit = st.number_input("Credit Amount", value=500000.0)
amt_annuity = st.number_input("Annuity", value=25000.0)
ext_source_2 = st.number_input("External Source 2 Score", value=0.5)
ext_source_3 = st.number_input("External Source 3 Score", value=0.5)

# Create dataframe (must match training features you used)
input_data = pd.DataFrame({
    "AMT_INCOME_TOTAL": [amt_income],
    "AMT_CREDIT": [amt_credit],
    "AMT_ANNUITY": [amt_annuity],
    "EXT_SOURCE_2": [ext_source_2],
    "EXT_SOURCE_3": [ext_source_3]
})

# Predict
if st.button("Predict Risk"):
    prediction = model.predict_proba(input_data)[0][1]
    st.write(f"Risk Probability: {prediction:.2f}")

    if prediction > 0.5:
        st.error("High Risk Applicant ❌")
    else:
        st.success("Low Risk Applicant ✅")