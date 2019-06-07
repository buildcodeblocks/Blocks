from sklearn.svm import SVR

# support vector regressor


def supportVectorRegressor(X, y, kernel="rbf", degree=3, C=1, tol=0.0001, epsilon=0.1):
    model = SVR(kernel, degree, C, tol, epsilon)
    model.fit(X, y)
    return model
