#!/usr/bin/env python3
"""
Task 5
"""
from sklearn import linear_model


def Logistic_Regression_Model(random_state):
    """
    Creates and returns an untrained LogisticRegression model instance
    Parameters:
        random_state (int): Seed for the random number generator.

    Returns:
        model: An untrained LogisticRegression instance.
    """
    model = linear_model.LogisticRegression(random_state=random_state)
    return model
