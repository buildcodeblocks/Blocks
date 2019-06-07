from sklearn.tree import DecisionTreeClassifier

# Decision Tree Classifier


def decisionTree(X, y, criterion="gini", max_depth=None):
    dt = DecisionTreeClassifier(criterion, max_depth)
    dt.fit(X, y)
    return dt
