"""
@author: K. Kersting, Z. Yu, J.Czech
Machine Learning Group, TU Darmstadt
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def fit_tree_stump_forest(X_train: np.ndarray, y_train: np.ndarray, n_estimators: int) -> RandomForestClassifier:
    """Creates and fits a random forrest of tree stumps on the training data."""
    raise NotImplementedError


def fit_tree_stump(X_train: np.ndarray, y_train: np.ndarray) -> tree.DecisionTreeClassifier:
    """Creates and fits a tree stump on the training data."""
    raise NotImplementedError


def main():
    # for reproducibility
    np.random.seed(42)

    # load data
    X_data = np.loadtxt(open('./PtU/FileName_Fz_raw.csv', 'r'), delimiter=",", skiprows=0)
    y_data = np.loadtxt(open('./PtU/FileName_Speed.csv', 'r'), delimiter=",", skiprows=0)
    print("X_data.shape:", X_data.shape)
    print("y_data.shape:", y_data.shape)

    # down sample the data
    X_sample = X_data[:, ::100]

    # split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_sample, y_data, test_size=0.2)

    # train
    clf = fit_tree_stump_forest(X_train, y_train, n_estimators=100)
    # predict
    y_pred = clf.predict(X_test)
    print("y_pred:", y_pred[:10], "...")
    print("y_test:", y_test[:10], '...')

    # show confusion matrix
    print("Train Confusion Matrix:\n", confusion_matrix(y_train, clf.predict(X_train)))
    print("Test Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # evaluate
    acc = accuracy_score(y_test, y_pred)
    print('Random Forest Test Accuracy:', acc)

    # compare with decision tree, max_depth=1, cannot classify 3-class data
    clf = fit_tree_stump(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print('Tree Stump Tree Test Accuracy:', acc)


if __name__ == '__main__':
    main()
