import os
import time
import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import confusion_matrix, classification_report

from model.preprocess import preprocess_test_data
from model.metrics import evaluate_classification_model

MODELS_DIR = os.path.join("model", "saved_models")

# Mapping of display names to saved model filenames
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
}

# ===============================
# Page Configuration
# ===============================
st.set_page_config(
    page_title="Phishing Website Detection",
    layout="centered"
)

st.title("Phishing Website Detection")
st.write("Machine Learning Assignment 2 - Model Evaluation Dashboard")
st.markdown("---")

# ===============================
# Dataset Upload
# ===============================

# Provide a quick download link for the test dataset
test_data_path = os.path.join("data", "test_data.csv")
if os.path.exists(test_data_path):
    with open(test_data_path, "rb") as f:
        st.download_button(
            label="Download Test Dataset (test_data.csv)",
            data=f,
            file_name="test_data.csv",
            mime="text/csv",
        )

uploaded_file = st.file_uploader(
    "Upload Test Dataset (CSV)",
    type=["csv"]
)

# ===============================
# Model Selection
# ===============================
model_name = st.selectbox(
    "Select Machine Learning Model",
    list(MODEL_FILES.keys())
)

st.markdown("---")

# ===============================
# Run Model
# ===============================
if st.button("Run Model") and uploaded_file is not None:

    start_time = time.time()

    # Preprocess test data using saved artifacts
    X_test, y_test = preprocess_test_data(uploaded_file, target_column="status")
    preprocess_time = time.time() - start_time

    # Load the pre-trained model
    model_path = os.path.join(MODELS_DIR, MODEL_FILES[model_name])
    model = joblib.load(model_path)
    load_time = time.time() - start_time - preprocess_time

    # ===============================
    # Predictions
    # ===============================
    predict_start = time.time()
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = None
    predict_time = time.time() - predict_start

    total_time = time.time() - start_time

    metrics = evaluate_classification_model(y_test, y_pred, y_prob)

    # ===============================
    # Timing Log
    # ===============================
    st.info(
        f"Preprocessing: {preprocess_time:.3f}s | "
        f"Model load: {load_time:.3f}s | "
        f"Prediction ({len(y_test)} samples): {predict_time:.3f}s | "
        f"**Total: {total_time:.3f}s**"
    )

    # ===============================
    # Evaluation Metrics Display
    # ===============================
    st.markdown("## Key Evaluation Metrics")

    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    col1.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
    auc_val = metrics['AUC']
    col2.metric("AUC", f"{auc_val:.3f}" if auc_val != "N/A" else "N/A")
    col3.metric("Precision", f"{metrics['Precision']:.3f}")

    col4.metric("Recall", f"{metrics['Recall']:.3f}")
    col5.metric("F1 Score", f"{metrics['F1 Score']:.3f}")
    col6.metric("MCC", f"{metrics['MCC']:.3f}")

    st.markdown("---")

    # ===============================
    # Confusion Matrix
    # ===============================
    st.markdown("## Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Legitimate", "Actual Phishing"],
        columns=["Predicted Legitimate", "Predicted Phishing"]
    )
    st.dataframe(cm_df)

    # ===============================
    # Classification Report
    # ===============================
    st.markdown("## Classification Report")

    report_df = pd.DataFrame(
        classification_report(
            y_test,
            y_pred,
            target_names=["Legitimate", "Phishing"],
            output_dict=True
        )
    ).transpose().round(3)

    st.dataframe(report_df)
