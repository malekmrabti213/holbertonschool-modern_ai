#!/usr/bin/env python3
"""
Task 1
"""


def train_tree(clf, X, y):
    """
    Trains the given DecisionTreeClassifier on the full dataset.

    Args:
        clf: A Scikit-learn classifier instance.
        X: Feature matrix.
        y: Target labels.

    Returns:
        None
    """
    clf.fit(X, y)
