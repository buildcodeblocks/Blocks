from sklearn.ensemble import RandomForestClassifier

# Random Forest Classifier


def randomForest(X, y, n_estimators=10, max_depth=None):
    rf = RandomForestClassifier(n_estimators, max_depth)
    rf.fit(X, y)
    return rf
