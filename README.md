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

### Logistic Regression

Logistic Regression achieved good performance with **87% accuracy and 0.92 AUC**, providing a strong baseline model. It maintained balanced precision and recall (0.84), indicating stable predictions. However, its performance is slightly lower than tree-based models due to its limitation in capturing non-linear relationships.

---

### Decision Tree

Decision Tree performed significantly better with **94% accuracy and 0.94 AUC**, along with strong precision and recall (~0.93–0.94). It effectively captured non-linear patterns in the data, resulting in a high F1 score and MCC (0.89). However, it may still be prone to slight overfitting.

---

### K-Nearest Neighbors (KNN)

KNN showed the weakest performance with **74% accuracy and 0.79 AUC**. Lower recall (0.66) and MCC (0.47) indicate poor generalization. This is likely due to high dimensionality and large dataset size, where distance-based methods become less effective.

---

### Naive Bayes

Naive Bayes performed comparably to Logistic Regression with **87% accuracy and 0.92 AUC**. It maintained balanced precision and recall (~0.82–0.84), but its strong independence assumption limits its ability to capture feature interactions, resulting in slightly lower performance than tree-based models.

---

### Random Forest (Ensemble)

Random Forest achieved excellent results with **96% accuracy and 0.99 AUC**, along with very high precision (0.97) and F1 score (0.96). It effectively reduces overfitting and captures complex feature interactions, leading to a high MCC (0.92), making it one of the most robust models.

---

### XGBoost (Ensemble)

XGBoost delivered the best overall performance with **96% accuracy, 0.99 AUC, and highest MCC (0.93)**. It slightly outperformed Random Forest in recall (0.95), indicating better detection of positive cases. Its boosting mechanism allows it to capture complex patterns and generalize well on large datasets.

---

Overall, **ensemble models (Random Forest and XGBoost)** clearly outperform traditional models. Among them, **XGBoost is the best-performing model**, achieving the highest overall balance across all evaluation metrics.

---

## Author

Muskaan Mittal
2025AA05875
M.Tech (AIML / DSE)  
BITS Pilani – Work Integrated Learning Programme
