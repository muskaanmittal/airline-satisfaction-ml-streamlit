# Airline Passenger Satisfaction Prediction using Machine Learning

## a. Problem Statement

The objective of this project is to build and compare multiple machine learning classification models to predict whether an airline passenger is satisfied or neutral/dissatisfied based on their travel experience and service ratings.

The project implements six different classification algorithms and evaluates their performance using standard classification metrics. An interactive Streamlit web application is also developed to allow users to upload test data, select a model, and view predictions and evaluation metrics.

---

## b. Dataset Description

The dataset used for this project is the Airline Passenger Satisfaction Dataset obtained from Kaggle.

Dataset Source:  
https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction

The dataset contains passenger feedback and flight details such as:

- Age
- Gender
- Type of Travel
- Class
- Flight Distance
- Inflight service ratings (seat comfort, food, wifi, cleanliness, etc.)
- Delay information

Target Variable:

- satisfaction
  - Satisfied → 1
  - Neutral or Dissatisfied → 0

Dataset Size:

- Total Instances: ~103,900
- Total Features: 24

This is a binary classification problem.

---

## c. Models Used and Evaluation Metrics

The following six classification models were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (KNN) Classifier
4. Naive Bayes Classifier
5. Random Forest (Ensemble Model)
6. XGBoost (Ensemble Model)

Each model was evaluated using the following metrics:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

## Model Comparison Table

| ML Model                 | Accuracy | AUC  | Precision | Recall | F1 Score | MCC  |
| ------------------------ | -------- | ---- | --------- | ------ | -------- | ---- |
| Logistic Regression      | 0.87     | 0.92 | 0.84      | 0.84   | 0.84     | 0.72 |
| Decision Tree            | 0.94     | 0.94 | 0.93      | 0.94   | 0.94     | 0.89 |
| KNN                      | 0.74     | 0.79 | 0.72      | 0.66   | 0.69     | 0.47 |
| Naive Bayes              | 0.87     | 0.92 | 0.82      | 0.82   | 0.84     | 0.72 |
| Random Forest (Ensemble) | 0.96     | 0.99 | 0.97      | 0.94   | 0.96     | 0.92 |
| XGBoost (Ensemble)       | 0.96     | 0.99 | 0.97      | 0.95   | 0.96     | 0.93 |

---

## Model Performance Observations

| ML Model Name | Observation about model performance |
| **Logistic Regression** | Logistic Regression provided a strong baseline model with moderate accuracy. It performs well on large datasets and gives stable predictions, but its performance is limited because it assumes a linear decision boundary and cannot fully capture complex non-linear relationships in the data. |
| **Decision Tree** | Decision Tree achieved high accuracy and good precision-recall balance. It is easy to interpret and captures non-linear patterns well, but it may slightly overfit compared to ensemble models. |
| **KNN** | KNN performed the weakest among all models due to the large dataset size and high dimensionality. Distance-based learning becomes inefficient in high-dimensional spaces, which leads to lower accuracy and poor generalization. |
| **Naive Bayes** | Naive Bayes achieved reasonable accuracy and fast training time. However, its strong independence assumption between features limits its performance compared to tree-based and ensemble models. |
| **Random Forest (Ensemble)** | Random Forest achieved excellent performance with very high accuracy and AUC. It effectively handles feature interactions and reduces overfitting by combining multiple decision trees, making it a robust and reliable model. |
| **XGBoost (Ensemble)** | XGBoost delivered the best overall performance with the highest accuracy, AUC, and MCC score. Its boosting mechanism efficiently captures complex patterns and provides superior generalization on large datasets. |

---

## Author

Muskaan Mittal
2025AA05875
M.Tech (AIML / DSE)  
BITS Pilani – Work Integrated Learning Programme
