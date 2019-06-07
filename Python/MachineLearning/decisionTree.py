from sklearn.tree import DecisionTreeClassifier

# Decision Tree Classifier


def decisionTree(X, y, criterion="gini", max_depth=None):
    model = DecisionTreeClassifier(criterion, max_depth)
    model.fit(X, y)
    return model
