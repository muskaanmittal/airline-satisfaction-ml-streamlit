import streamlit as st
import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

st.title("Airline Passenger Satisfaction Classification")

uploaded_file = st.file_uploader("Upload test CSV file", type=["csv"])

model_name = st.selectbox(
    "Select Model",
    ["Logistic", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
)

model_files = {
    "Logistic": "model/logistic.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
    "XGBoost": "model/xgboost.pkl"
}

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # ---------------- PREPROCESS (SAME AS TRAINING) ----------------

    # Target
    y_true = df['satisfaction'].map({
        'satisfied': 1,
        'neutral or dissatisfied': 0
    })

    df = df.drop(columns=['satisfaction', 'id', 'Unnamed: 0'], errors='ignore')

    # Encode categorical features
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(df)

    # ---------------- LOAD MODEL ----------------
    with open(model_files[model_name], "rb") as f:
        model = pickle.load(f)

    # ---------------- PREDICT ----------------
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # ---------------- DISPLAY METRICS ----------------
    st.subheader("Evaluation Metrics")
    st.write("Accuracy:", accuracy_score(y_true, y_pred))
    st.write("AUC:", roc_auc_score(y_true, y_prob))
    st.write("Precision:", precision_score(y_true, y_pred))
    st.write("Recall:", recall_score(y_true, y_pred))
    st.write("F1 Score:", f1_score(y_true, y_pred))
    st.write("MCC:", matthews_corrcoef(y_true, y_pred))

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y_true, y_pred))
