from sklearn.linear_model import LogisticRegression

# Logistic Regression Model


def logisticRegression(X, y, penalty="l2", C=1, fit_intercept=True):
    model = LogisticRegression(penalty, C, fit_intercept)
    model.fit(X, y)
    return model
