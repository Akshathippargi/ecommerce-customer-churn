import streamlit as st
import pandas as pd
import pickle
import os

# -----------------------
# Page Configuration
# -----------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="centered"
)

st.title("📉 Customer Churn Prediction Dashboard")
st.write(
    """
    Predict the likelihood of a customer churning based on
    historical purchase behavior.
    """
)

# -----------------------
# Load Model (cached)
# -----------------------
@st.cache_resource
def load_model():
    model_path = os.path.join("models", "xgboost_churn_model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# -----------------------
# Sidebar Inputs
# -----------------------
st.sidebar.header("Customer Inputs")

frequency = st.sidebar.number_input(
    "Purchase Frequency",
    min_value=0,
    step=1,
    help="Number of invoices placed by the customer"
)

total_units = st.sidebar.number_input(
    "Total Units Purchased",
    min_value=0,
    step=1
)

monetary_value = st.sidebar.number_input(
    "Total Monetary Value",
    min_value=0.0,
    step=50.0
)

# Feature dataframe
input_df = pd.DataFrame([{
    "frequency": frequency,
    "total_units": total_units,
    "monetary_value": monetary_value
}])

# -----------------------
# Prediction
# -----------------------
st.markdown("## 🔍 Prediction")

if st.button("Predict Churn Risk"):
    prob = model.predict_proba(input_df)[0][1]

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Churn Probability", f"{prob:.2%}")

    with col2:
        if prob >= 0.6:
            st.error("🔴 High Risk")
        elif prob >= 0.3:
            st.warning("🟠 Medium Risk")
        else:
            st.success("🟢 Low Risk")

    st.info(
        "💡 Tip: Customers with low purchase frequency and low lifetime value "
        "are more likely to churn."
    )

st.markdown("---")
st.caption(
    "Model: XGBoost | Features: Frequency, Total Units, Monetary Value"
)
