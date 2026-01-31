import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import *

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

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    y_true = df['satisfaction']
    X = df.drop('satisfaction', axis=1)

    with open(model_files[model_name], "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    st.write("Accuracy:", accuracy_score(y_true, y_pred))
    st.write("AUC:", roc_auc_score(y_true, y_prob))
    st.write("Precision:", precision_score(y_true, y_pred))
    st.write("Recall:", recall_score(y_true, y_pred))
    st.write("F1:", f1_score(y_true, y_pred))
    st.write("MCC:", matthews_corrcoef(y_true, y_pred))

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y_true, y_pred))
