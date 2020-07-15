"""
@author: K. Kersting, Z. Yu, J.Czech
Machine Learning Group, TU Darmstadt
"""
import graphviz
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def fit_dt_classifier(X_train: np.ndarray, y_train: np.ndarray) -> tree.DecisionTreeClassifier:
    """Creates and fits a decision tree classifier on the training data."""
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    return clf


def get_test_accuracy(clf, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """Evaluates the test accuracy for a given classifier and test dataset."""
    pre = clf.predict(X_test)
    acc = accuracy_score(y_test, pre)
    return acc


def export_tree_plot(clf, filename: str) -> None:
    """Exports the tree plot for the given classifier as a pdf with given filename."""
    dot_data = tree.export_graphviz(clf, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render(filename)


def main():
    # for reproducibility
    np.random.seed(42)

    # load data
    X_data = np.loadtxt(open('./Data/FileName_Fz_raw.csv', 'r'), delimiter=",", skiprows=0)
    y_data = np.loadtxt(open('./Data/FileName_Speed.csv', 'r'), delimiter=",", skiprows=0)

    # down sample the data
    X_sample = X_data[:, ::100]
    print("X_data.shape:", X_data.shape)
    print("y_data.shape:", y_data.shape)

    # split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_sample, y_data, test_size=0.2)

    # train
    clf = fit_dt_classifier(X_train, y_train)
    # predict
    acc = get_test_accuracy(clf, X_test, y_test)
    print('Test Accuracy:', acc)

    print("predict_proba:", clf.predict_proba(X_test))

    # plot tree
    export_tree_plot(clf, "classification_tree")


if __name__ == '__main__':
    main()


