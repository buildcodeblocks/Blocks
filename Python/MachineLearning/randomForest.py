from sklearn.ensemble import RandomForestClassifier

# Random Forest Classifier


def randomForest(X, y, n_estimators=10, max_depth=None):
    model = RandomForestClassifier(n_estimators, max_depth)
    model.fit(X, y)
    return model
