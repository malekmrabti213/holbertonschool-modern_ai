#!/usr/bin/env python3
"""
Task 0
"""
from sklearn.tree import DecisionTreeClassifier


def build_decision_tree(min_samples_leaf, min_samples_split, random_state):
    """
    
    """
    clf = DecisionTreeClassifier(criterion='gini', max_depth=None,
                                 min_samples_leaf=min_samples_leaf,
                                 min_samples_split=min_samples_split,
                                 random_state=random_state)
    return clf
