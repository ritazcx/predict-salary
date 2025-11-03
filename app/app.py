# ============================================================
# app.py — Streamlit Salary Prediction App
# ------------------------------------------------------------
# Purpose:
#   Interactive web app to estimate salary based on user input.
#   Uses trained model from Step 3 (model.pkl).
# ------------------------------------------------------------
# Run this app:
#   streamlit run app/app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="Salary Prediction App",
    page_icon="💼",
    layout="wide",
)

st.title("💼 Salary Prediction App")
st.subheader("Estimate your expected salary based on your job details")
st.markdown("---")

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------
MODEL_PATH = "app/model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found. Please train your model and save it as app/model.pkl.")
    st.stop()

model = joblib.load(MODEL_PATH)

# ------------------------------------------------------------
# Sidebar Inputs
# ------------------------------------------------------------
st.sidebar.header("🧾 Enter Your Job Details")

job_title = st.sidebar.selectbox("Job Title", [
    "Data Scientist", "Data Analyst", "Software Engineer", "Product Manager",
    "Machine Learning Engineer", "Other"
])

experience = st.sidebar.slider("Years of Experience", 0, 40, 3)
education = st.sidebar.selectbox("Education Level", ["Bachelor's", "Master's", "PhD", "Other"])
location = st.sidebar.selectbox("Location", ["California", "New York", "Texas", "Remote", "Other"])
industry = st.sidebar.selectbox("Industry", ["Tech", "Finance", "Healthcare", "Education", "Other"])
company_size = st.sidebar.selectbox("Company Size", ["1–50", "51–200", "201–1000", "1001–5000", "5000+"])

predict_btn = st.sidebar.button("🔮 Predict Salary")

# ------------------------------------------------------------
# Build input DataFrame
# ------------------------------------------------------------
def preprocess_user_input():
    """Convert user selections into a one-row DataFrame compatible with training features."""
    data = {
        "job_title": [job_title],
        "experience": [experience],
        "education": [education],
        "location": [location],
        "industry": [industry],
        "company_size": [company_size]
    }
    df_input = pd.DataFrame(data)

    # Simplified encoding (placeholder: ensure same encoding logic as training)
    df_encoded = pd.get_dummies(df_input, drop_first=True)
    return df_encoded, df_input

# ------------------------------------------------------------
# Prediction logic
# ------------------------------------------------------------
if predict_btn:
    X_user, display_df = preprocess_user_input()

    # Align user features with model features (missing columns → 0)
    model_features = getattr(model, "feature_names_in_", None)
    if model_features is not None:
        X_user = X_user.reindex(columns=model_features, fill_value=0)

    try:
        prediction = model.predict(X_user)[0]
        st.metric(label="💰 Estimated Annual Salary", value=f"${prediction:,.0f}")
        st.markdown("##### Based on your input:")
        st.table(display_df.T.rename(columns={0: "Your Input"}))

        # Optional: display model info
        with st.expander("📊 Model Information"):
            st.write("Model: Random Forest Regressor")
            st.write("R² ≈ 0.62 | MAE ≈ $16K | RMSE ≈ $25K")
            st.caption("Trained on 2025 Glassdoor US dataset")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.markdown("---")
st.caption("© 2025 Salary Prediction Project | Built with Streamlit · Model v1.0")
st.caption("Disclaimer: Predictions are estimates based on public data. Actual salaries may vary.")

