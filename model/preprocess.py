import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "saved_models")


def load_and_prepare_data(csv_path, target_column):
    """
    Loads dataset, preprocesses features, and returns train-test split.
    Also saves the fitted scaler, imputer, and label encoder for later use.
    """

    # Load dataset
    df = pd.read_csv(csv_path)

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encode target labels (legitimate / phishing)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Drop non-numeric columns (URL text)
    X = X.select_dtypes(include=["int64", "float64"])
    feature_columns = list(X.columns)

    # Handle missing values
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)

    # Feature scaling
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Save preprocessing artifacts
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(label_encoder, os.path.join(ARTIFACTS_DIR, "label_encoder.pkl"))
    joblib.dump(imputer, os.path.join(ARTIFACTS_DIR, "imputer.pkl"))
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
    joblib.dump(feature_columns, os.path.join(ARTIFACTS_DIR, "feature_columns.pkl"))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def preprocess_test_data(csv_path, target_column):
    """
    Loads test data and transforms it using saved preprocessing artifacts.
    Returns X_test (scaled features) and y_test (encoded labels).
    """

    df = pd.read_csv(csv_path)

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Load saved artifacts
    label_encoder = joblib.load(os.path.join(ARTIFACTS_DIR, "label_encoder.pkl"))
    imputer = joblib.load(os.path.join(ARTIFACTS_DIR, "imputer.pkl"))
    scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
    feature_columns = joblib.load(os.path.join(ARTIFACTS_DIR, "feature_columns.pkl"))

    # Encode target
    y = label_encoder.transform(y)

    # Keep only the numeric columns used during training
    X = X[feature_columns]

    # Transform using fitted imputer and scaler
    X = imputer.transform(X)
    X = scaler.transform(X)

    return X, y
