#!/usr/bin/env python3
"""
Task 0
"""
from sklearn import linear_model


def Linear_Regression():
    """
    Creates and returns an untrained LinearRegression model instance.

    The model uses ordinary least squares to fit a linear model to the data.
    """
    return linear_model.LinearRegression()
