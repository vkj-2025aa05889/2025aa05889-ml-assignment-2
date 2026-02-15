from sklearn.neighbors import KNeighborsClassifier


def train_knn_classifier(X_train, y_train):
    model = KNeighborsClassifier(
        n_neighbors=7,
        weights="distance"
    )
    model.fit(X_train, y_train)
    return model
