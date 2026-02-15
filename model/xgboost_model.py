from xgboost import XGBClassifier


def train_xgboost_classifier(X_train, y_train):
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)
    return model
