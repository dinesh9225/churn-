import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

# Load dataset and preprocess with caching
@st.cache_data
def load_data():
    df = pd.read_csv("customer_churn.csv")
    df.fillna(df.median(numeric_only=True), inplace=True)
    le = LabelEncoder()
    for col in ['ContractType', 'HasInternet', 'IsPremium']:
        df[col] = le.fit_transform(df[col])
    return df

df = load_data()

st.title("Customer Churn Prediction")
st.write("## Dataset Preview")
st.write(df.tail())
st.write(f"Dataset Shape: {df.shape}")

# Prepare features and target
X = df.drop(['CustomerID', 'Churn'], axis=1)
y = df['Churn']

# Load saved scaler and models
scaler = joblib.load("scaler.pkl")
rf_model = joblib.load("random_forest_churn.pkl")
lr_model = joblib.load("logistic_regression_churn.pkl")

# Sidebar inputs for customer data
st.sidebar.header("Input Customer Data")

input_dict = {}
for feature in X.columns:
    if np.issubdtype(X[feature].dtype, np.number):
        min_val = float(X[feature].min())
        max_val = float(X[feature].max())
        mean_val = float(X[feature].mean())
        input_dict[feature] = st.sidebar.number_input(f"{feature}", min_value=min_val, max_value=max_val, value=mean_val)
    else:
        input_dict[feature] = st.sidebar.selectbox(f"{feature}", options=X[feature].unique())

input_df = pd.DataFrame([input_dict])

# Scale inputs
input_scaled = scaler.transform(input_df)

# Predict on button click
if st.sidebar.button("Predict Churn"):
    rf_pred = rf_model.predict(input_scaled)[0]
    rf_prob = rf_model.predict_proba(input_scaled)[0][1]

    lr_pred = lr_model.predict(input_scaled)[0]
    lr_prob = lr_model.predict_proba(input_scaled)[0][1]

    churn_dict = {0: "No", 1: "Yes"}

    st.write("### Prediction Results")
    st.write(f"Random Forest Prediction: {churn_dict[rf_pred]} (Probability: {rf_prob:.2f})")
    st.write(f"Logistic Regression Prediction: {churn_dict[lr_pred]} (Probability: {lr_prob:.2f})")

# Show Model Performance if checked
if st.checkbox("Show Model Performance"):
    X_scaled = scaler.transform(X)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

    st.write("## Random Forest Model Performance")
    y_pred_rf = rf_model.predict(X_test)
    st.text(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_rf)}")
    st.text(f"Classification Report:\n{classification_report(y_test, y_pred_rf)}")
    st.text(f"ROC AUC: {roc_auc_score(y_test, rf_model.predict_proba(X_test)[:,1]):.3f}")

    st.write("## Logistic Regression Model Performance")
    y_pred_lr = lr_model.predict(X_test)
    st.text(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_lr)}")
    st.text(f"Classification Report:\n{classification_report(y_test, y_pred_lr)}")
    st.text(f"ROC AUC: {roc_auc_score(y_test, lr_model.predict_proba(X_test)[:,1]):.3f}")
