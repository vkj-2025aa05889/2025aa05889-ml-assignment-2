from model.preprocess import load_and_prepare_data

X_train, X_test, y_train, y_test = load_and_prepare_data(
    "data/dataset_phishing.csv",
    target_column="status"
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Unique labels:", set(y_train))
