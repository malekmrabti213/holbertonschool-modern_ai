#!/usr/bin/env python3
"""
Task 2
"""
from sklearn import linear_model


def ridge_regression(random_state):
    """
    Creates and returns an untrained RidgeRegression model instance.
    """
    return linear_model.Ridge(random_state=random_state)
