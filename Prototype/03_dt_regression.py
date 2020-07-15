"""
Created on January 25, 2018
@author: K. Kersting, Z. Yu, J.Czech
Machine Learning Group, TU Darmstadt
"""
from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np
import graphviz
from sklearn.metrics import mean_squared_error


def fit_dt_regressor(X_train: np.ndarray, y_train: np.ndarray, max_depth=None) -> tree.DecisionTreeRegressor:
    """Creates and fits a regression tree on the training data."""
    raise NotImplementedError


def get_test_mse(clf, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """Evaluates the test mse for a given classifier and test dataset."""
    raise NotImplementedError


def export_tree_plot(clf, filename: str) -> None:
    """Exports the tree plot for the given classifier as a pdf with given filename."""
    raise NotImplementedError


def main():
    # for reproducibility
    np.random.seed(42)

    # load data
    X_data = np.loadtxt(open('./PtU/FileName_Fz_raw.csv', 'r'), delimiter=",", skiprows=0)
    y_data = np.loadtxt(open('./PtU/FileName_thickness.csv', 'r'), delimiter=",", skiprows=0)
    print("X_data.shape:", X_data.shape)
    print("y_data.shape:", y_data.shape)

    # down sample the data
    X_sample = X_data[:, ::100]

    # split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_sample, y_data, test_size=0.2)

    # train
    clf = fit_dt_regressor(X_train, y_train)
    # predict % evaluate
    mse = get_test_mse(clf, X_test, y_test)
    print('Test MSE:', mse)

    # change max tree depth
    # train
    clf = fit_dt_regressor(X_train, y_train, max_depth=3)

    # predict & evaluate
    mse = get_test_mse(clf, X_test, y_test)
    print('Test MSE:', mse)

    # plot tree
    export_tree_plot(clf, "regression_tree_d3")


if __name__ == '__main__':
    main()
