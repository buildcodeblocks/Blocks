from sklearn.linear_model import LinearRegression

# linear regression model


def linearRegression(X, y, fit_intercept=True):
    reg = LinearRegression(fit_intercept)
    reg.fit(X, y)
    return reg
