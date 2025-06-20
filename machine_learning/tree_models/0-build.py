#!/usr/bin/env python3
"""
Task 0
"""
from sklearn.tree import DecisionTreeClassifier


def build_decision_tree(min_samples_leaf, min_samples_split, random_state):
    """
    Builds and returns a DecisionTreeClassifier with the parameters.

    The classifier uses:
    - Gini impurity as the criterion for split quality
    - No maximum depth limit (max_depth=None)
    - Custom values for min_samples_leaf and min_samples_split
    - A fixed random_state for reproducibility

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
