from sklearn.svm import SVC

# SVC classifier


def supportVectorClassifier(X, y, kernel="rbf", penalty="l2", degree=3, C=1, tol=0.0001):
    model = SVC(kernel, penalty, degree, C, tol)
    model.fit(X, y)
    return model
