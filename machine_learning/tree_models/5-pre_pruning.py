#!/usr/bin/env python3
"""
Task 5
"""
from sklearn import model_selection


def prepruning(X, y, clf):
    """
    Parameters:
        X_train (ndarray): Training features
        y_train (ndarray): Training labels
        seed (int): Random seed for reproducibility

    Returns:
        best_params (dict): Best hyperparameters found by GridSearchCV
    """

    grid_param = {
        "criterion": ["gini", "entropy"],
        "max_depth": range(2, 5),
        "min_samples_leaf": range(2, 5),
        "min_samples_split": range(2, 5)
    }
    grid_search = model_selection.GridSearchCV(estimator=clf,
                                               param_grid=grid_param)
    grid_search.fit(X, y)

    return grid_search.best_params_
