from lightgbm import LGBMClassifier

# light gradient boosting tree classifier fits faster than xgbm


def lightGBMClassifier(X, y, num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=100):
    model = LGBMClassifier(num_leaves, max_depth,
                           learning_rate, n_estimators)
    model.fit(X, y)
    return model
