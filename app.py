import streamlit as st
import numpy as np
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Loan Approval Prediction", page_icon="💰")

st.title("💰 Loan Approval Prediction")
st.write("Simple ML model running directly inside app")

# -------- Train model inside app (SAFE) --------
X = np.array([
    [20000, 5000],
    [30000, 7000],
    [50000, 15000],
    [80000, 30000]
])
y = np.array([0, 0, 1, 1])

model = LogisticRegression()
model.fit(X, y)

# -------- Inputs --------
income = st.number_input("Applicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)

# -------- Prediction --------
if st.button("Predict"):
    data = np.array([[income, loan_amount]])
    result = model.predict(data)

    if result[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")