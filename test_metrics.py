from model.metrics import evaluate_classification_model

y_true = [0, 1, 0, 1, 1]
y_pred = [0, 1, 0, 0, 1]
y_prob = [0.2, 0.8, 0.3, 0.4, 0.9]

metrics = evaluate_classification_model(y_true, y_pred, y_prob)
print(metrics)
