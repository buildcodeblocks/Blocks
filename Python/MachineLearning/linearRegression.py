from sklearn.linear_model import LinearRegression

# linear regression model


def linearRegression(X, y, fit_intercept=True):
    model = LinearRegression(fit_intercept)
    model.fit(X, y)
    return model
