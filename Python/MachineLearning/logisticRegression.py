from sklearn.linear_model import LogisticRegression

# Logistic Regression Model


def logisticRegression(X, y, penalty="l2", C=1, fit_intercept=True):
    reg = LogisticRegression(penalty, C, fit_intercept)
    reg.fit(X, y)
    return reg
