#!/usr/bin/env python3
"""
Task 6
"""


def get_pruning_path(clf, X, y):
    """
    Parameters:
        clf (DecisionTreeClassifier): A decision tree classifier instance
        X (ndarray or DataFrame): Feature matrix.
        y (ndarray or Series): Target labels.

    Returns:
        ccp_alphas (ndarray): Effective alpha values for pruning.
        impurities (ndarray): Total impurity of leaves at each alpha.
    """
    path = clf.cost_complexity_pruning_path(X, y)
    return path.ccp_alphas, path.impurities
