from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef
)


def evaluate_classification_model(y_true, y_pred, y_prob=None):
    """
    Calculates evaluation metrics for a classification model.
    """

    results = {}

    results["Accuracy"] = accuracy_score(y_true, y_pred)
    results["Precision"] = precision_score(y_true, y_pred, average="weighted")
    results["Recall"] = recall_score(y_true, y_pred, average="weighted")
    results["F1 Score"] = f1_score(y_true, y_pred, average="weighted")
    results["MCC"] = matthews_corrcoef(y_true, y_pred)

    if y_prob is not None:
        results["AUC"] = roc_auc_score(y_true, y_prob)
    else:
        results["AUC"] = "N/A"

    return results
