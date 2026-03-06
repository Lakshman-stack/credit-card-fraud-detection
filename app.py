import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load("fraud_model.pkl")

st.set_page_config(page_title="Fraud Detection", layout="wide")

st.title("💳 AI Credit Card Fraud Detection System")

st.markdown("""
This system uses **Machine Learning (Random Forest)** to detect fraudulent credit card transactions.

Dataset size: **284,807 transactions**  
Fraud cases: **492**
""")

st.divider()

# -----------------------------
# Transaction Input Section
# -----------------------------

st.header("Enter Transaction Details")

# Create feature names
feature_names = ["Amount"] + [f"V{i}" for i in range(1,29)]

user_inputs = []

for feature in feature_names:
    value = st.number_input(feature, value=0.0)
    user_inputs.append(value)

st.divider()

# -----------------------------
# Prediction
# -----------------------------

if st.button("Check Transaction"):

    features = np.array([user_inputs])

    prediction = model.predict(features)
    probability = model.predict_proba(features)

    fraud_prob = probability[0][1] * 100

    st.subheader("Prediction Result")

    st.write("Fraud Probability:", round(fraud_prob,2), "%")

    if prediction[0] == 1:
        st.error("⚠ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Normal Transaction")

st.divider()

# -----------------------------
# Fraud Dataset Statistics
# -----------------------------

st.header("Fraud Dataset Statistics")

chart_data = pd.DataFrame({
    "Transaction Type": ["Normal", "Fraud"],
    "Count": [284315, 492]
})

st.bar_chart(chart_data.set_index("Transaction Type"))

st.write("Fraud percentage in dataset:", round((492/284807)*100,4), "%")

st.divider()

# -----------------------------
# Feature Importance
# -----------------------------

st.header("Model Feature Importance")

importances = model.feature_importances_

fig, ax = plt.subplots(figsize=(8,6))

ax.barh(feature_names, importances)

ax.set_xlabel("Importance Score")
ax.set_title("Feature Importance")

st.pyplot(fig)
#streamlit run app.py