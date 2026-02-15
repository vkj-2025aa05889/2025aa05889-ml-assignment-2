"""
Training script: trains all 6 models on the full dataset and saves them as .pkl files.
Run this once before deploying the Streamlit app.
"""

import os
import joblib

from model.preprocess import load_and_prepare_data
from model.logistic_regression_model import train_logistic_regression
from model.decision_tree_model import train_decision_tree
from model.knn_classifier_model import train_knn_classifier
from model.naive_bayes_model import train_naive_bayes
from model.random_forest_model import train_random_forest
from model.xgboost_model import train_xgboost_classifier

SAVE_DIR = os.path.join("model", "saved_models")

# Load and preprocess data (also saves scaler/imputer/encoder artifacts)
X_train, X_test, y_train, y_test = load_and_prepare_data(
    "data/dataset_phishing.csv", target_column="status"
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

models = {
    "logistic_regression": train_logistic_regression,
    "decision_tree": train_decision_tree,
    "knn": train_knn_classifier,
    "naive_bayes": train_naive_bayes,
    "random_forest": train_random_forest,
    "xgboost": train_xgboost_classifier,
}

os.makedirs(SAVE_DIR, exist_ok=True)

for name, train_fn in models.items():
    print(f"Training {name}...")
    model = train_fn(X_train, y_train)
    path = os.path.join(SAVE_DIR, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"  Saved to {path}")

print("\nAll models and preprocessing artifacts saved.")
