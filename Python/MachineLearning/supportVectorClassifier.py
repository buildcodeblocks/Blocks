from sklearn.svm import SVC

# SVC classifier


def supportVectorClassifier(X, y, kernel="rbf", penalty="l2", degree=3, C=1, tol=0.0001):
    svc = SVC(kernel, penalty, degree, C, tol)
    svc.fit(X, y)
    return svc
