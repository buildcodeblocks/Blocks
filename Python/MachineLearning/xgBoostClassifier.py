from xgboost import XGBClassifier

# X Gradient Boosting Classifier


def xgBoostClassifier(X, y, max_depth=3, learning_rate=0.1, n_estimators=100):
    model = XGBClassifier(max_depth, learning_rate, n_estimators)
    model.fit(X, y)
    return model
