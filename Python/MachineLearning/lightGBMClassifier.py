from lightgbm import LGBMClassifier

# light gradient boosting tree classifier fits faster than xgbm


def lightGBMClassifier(X, y, num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=100):
    lgbm = LGBMClassifier(num_leaves, max_depth,
                          learning_rate, n_estimators)
    lgbm.fit(X, y)
    return lgbm
