#!/usr/bin/env python3
"""
Task 9
"""
from sklearn import ensemble


def random_forest(n_estimators, random_state):
    """
    Arguments:
        random_state (int): Seed for reproducibility
        n_estimators (int): Number of estimators in the ensemble

    Returns:
        RandomForestClassifier
    """

    rf = ensemble.RandomForestClassifier(n_estimators=n_estimators,
                                         random_state=random_state)
    return rf
