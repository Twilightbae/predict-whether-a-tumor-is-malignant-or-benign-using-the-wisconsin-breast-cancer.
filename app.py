import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================
# Load the trained NN model
# ==========================
nn_model = tf.keras.models.load_model("nn_breast_cancer_model.h5")

st.set_page_config(page_title="Breast Cancer Risk Prediction", layout="wide")

st.title("🩺 Breast Cancer Risk Prediction Dashboard")
st.markdown(
    """
    This app predicts whether a breast tumor is **benign (0)** or **malignant (1)**  
    using a Neural Network trained on the Wisconsin Breast Cancer Dataset.
    """
)

# --------------------------
# Upload patient data
# --------------------------
st.sidebar.header("Upload Patient Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📂 Uploaded Patient Data")
    st.write(df.head())

    # Predict
    predictions = nn_model.predict(df).ravel()
    risk_scores = (predictions * 100).round(2)
    pred_labels = (predictions >= 0.5).astype(int)

    df["Risk Score (%)"] = risk_scores
    df["Prediction"] = pred_labels

    st.subheader("🔎 Predictions")
    st.write(df)

    # Risk Distribution Plot
    st.subheader("📊 Risk Score Distribution")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.histplot(df["Risk Score (%)"], bins=20, kde=True, color="purple", ax=ax)
    ax.set_title("Distribution of Predicted Risk Scores")
    st.pyplot(fig)

    # Scatter Plot
    st.subheader("🧬 Patient Risk Visualization")
    fig2, ax2 = plt.subplots(figsize=(8,6))
    sns.scatterplot(
        x=range(len(df)),
        y="Risk Score (%)",
        hue="Prediction",
        palette={0:"blue", 1:"red"},
        data=df,
        ax=ax2
    )
    ax2.axhline(50, ls="--", c="black")
    ax2.set_title("Patient Risk Scores (0=Benign, 1=Malignant)")
    ax2.set_xlabel("Patient Index")
    ax2.set_ylabel("Risk Score (%)")
    st.pyplot(fig2)

    # Download results
    st.sidebar.subheader("📥 Download Predictions")
    csv_download = df.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button(
        label="Download CSV",
        data=csv_download,
        file_name="predictions.csv",
        mime="text/csv",
    )
else:
    st.info("👆 Upload a CSV file to get predictions")
