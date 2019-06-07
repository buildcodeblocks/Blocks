from sklearn.neighbors import KNeighborsClassifier

# K nearest neighbors classifier


def kNearestNeighbors(X, y, n_neighbors=5, weights="uniform"):
    model = KNeighborsClassifier(n_neighbors, weights)
    model.fit(X, y)
    return model
