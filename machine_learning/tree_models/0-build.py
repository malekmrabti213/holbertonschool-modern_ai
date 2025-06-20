#!/usr/bin/env python3
"""
Task 0
"""
from sklearn.tree import DecisionTreeClassifier


def build_decision_tree(min_samples_leaf, min_samples_split, random_state):
    """
    Arguments:
        min_samples_leaf (int): samples required at a leaf node.
        min_samples_split (int): samples required to split an internal node.
        random_state (int): Random seed for reproducibility.

    Returns:
        DecisionTreeClassifier: A configured instance of the Scikit-learn.
    """
    clf = DecisionTreeClassifier(criterion='gini', max_depth=None,
                                 min_samples_leaf=min_samples_leaf,
                                 min_samples_split=min_samples_split,
                                 random_state=random_state)
    return clf
