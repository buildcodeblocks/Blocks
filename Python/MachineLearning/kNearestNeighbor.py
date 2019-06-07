from sklearn.neighbors import KNeighborsClassifier

# K nearest neighbors classifier


def kNearestNeighbors(X, y, n_neighbors=5, weights="uniform"):
    knn = KNeighborsClassifier(n_neighbors, weights)
    knn.fit(X, y)
    return knn
