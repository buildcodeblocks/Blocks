from sklearn.svm import LinearSVC

# Linear SVC classifier
# different from the Support Vector Classifier as only has liner as its type


def LinearSVM(X, y, penalty="l2", C=1, tol=0.0001, fit_intercept=True):
    model = LinearSVC(penalty, C, tol, fit_intercept)
    model.fit(X, y)
    return model
