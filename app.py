import streamlit as st
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

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
st.set_page_config(
    page_title="Airline Passenger Satisfaction Prediction",
    layout="wide"
)

st.title("✈️ Airline Passenger Satisfaction Prediction")
st.write("Upload test CSV data, select a model, and view predictions with evaluation metrics.")


# -----------------------------
# TRAIN MODELS (NO PICKLE)
# -----------------------------
@st.cache_resource
def train_models():
    # 👉 CHANGE THIS PATH if needed
    train_df = pd.read_csv("train.csv")  # your training dataset

    X = train_df.drop("satisfaction", axis=1)
    y = train_df["satisfaction"]

    if y.dtype == object:
        y = y.map({
            "satisfied": 1,
            "neutral or dissatisfied": 0
        })

    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X.select_dtypes(include=["object"]).columns

    num_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, numerical_cols),
        ("cat", cat_pipeline, categorical_cols)
    ])

    models = {
        "Logistic Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000))
        ]),

        "Decision Tree": Pipeline([
            ("preprocessor", preprocessor),
            ("model", DecisionTreeClassifier())
        ]),

        "KNN": Pipeline([
            ("preprocessor", preprocessor),
            ("model", KNeighborsClassifier())
        ]),

        "Naive Bayes": Pipeline([
            ("preprocessor", preprocessor),
            ("model", GaussianNB())
        ]),
    }

    for name in models:
        models[name].fit(X, y)

    return models


models = train_models()


# -----------------------------
# Sidebar
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
        st.error("Uploaded CSV must contain 'satisfaction' column.")
    else:
        X_test = df.drop("satisfaction", axis=1)
        y_test = df["satisfaction"]

        if y_test.dtype == object:
            y_test = y_test.map({
                "satisfied": 1,
                "neutral or dissatisfied": 0
            })

        model = models[model_name]

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        st.subheader("📊 Evaluation Metrics")

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", round(accuracy, 4))
        col1.metric("AUC", round(auc, 4))

        col2.metric("Precision", round(precision, 4))
        col2.metric("Recall", round(recall, 4))

        col3.metric("F1 Score", round(f1, 4))
        col3.metric("MCC", round(mcc, 4))

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Satisfied", "Satisfied"],
            yticklabels=["Not Satisfied", "Satisfied"]
        )
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        st.pyplot(fig)

        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))

else:
    st.info("Please upload a test CSV file to start prediction.")
