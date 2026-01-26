import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Airline Passenger Satisfaction Prediction",
                   layout="wide")


# -----------------------------
# Title
# -----------------------------
st.title("✈️ Airline Passenger Satisfaction Prediction")
st.write("Upload test CSV data, select a trained model, and view predictions with evaluation metrics.")


# -----------------------------
# Load Trained Models
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": joblib.load(os.path.join(MODEL_DIR, "logistic_model.pkl")),
        "Decision Tree": joblib.load(os.path.join(MODEL_DIR, "dt_model.pkl")),
        "KNN": joblib.load(os.path.join(MODEL_DIR, "knn_model.pkl")),
        "Naive Bayes": joblib.load(os.path.join(MODEL_DIR, "nb_model.pkl")),
        "XGBoost": joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl")),
    }
    return models


models = load_models()


# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("App Controls")

model_name = st.sidebar.selectbox(
    "Select ML Model",
    list(models.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test Dataset (CSV only)",
    type=["csv"]
)


# -----------------------------
# Main Logic
# -----------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Test Dataset Preview")
    st.dataframe(df.head())

    if "satisfaction" not in df.columns:
        st.error("Uploaded CSV must contain 'satisfaction' column for evaluation.")
    else:
        # Split features & target
        X_test = df.drop("satisfaction", axis=1)
        y_test = df["satisfaction"]

        # Convert labels if needed
        if y_test.dtype == object:
            y_test = y_test.map({
                "satisfied": 1,
                "neutral or dissatisfied": 0
            })

        # Load selected model
        model = models[model_name]

        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # -----------------------------
        # Metrics
        # -----------------------------
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        st.subheader("📊 Evaluation Metrics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Accuracy", round(accuracy, 4))
            st.metric("AUC Score", round(auc, 4))

        with col2:
            st.metric("Precision", round(precision, 4))
            st.metric("Recall", round(recall, 4))

        with col3:
            st.metric("F1 Score", round(f1, 4))
            st.metric("MCC", round(mcc, 4))

        # -----------------------------
        # Confusion Matrix
        # -----------------------------
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Not Satisfied", "Satisfied"],
                    yticklabels=["Not Satisfied", "Satisfied"])
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(f"Confusion Matrix - {model_name}")

        st.pyplot(fig)

        # -----------------------------
        # Classification Report
        # -----------------------------
        st.subheader("Classification Report")

        report = classification_report(y_test, y_pred, output_dict=False)
        st.text(report)

else:
    st.info("Please upload a test CSV file from the sidebar to start prediction.")
