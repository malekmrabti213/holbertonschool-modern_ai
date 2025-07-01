#!/usr/bin/env python3
"""
Task 1
"""
from sklearn import metrics
import numpy as np


def evaluation_metrics_for_regression(y_true, y_pred):
    """
    Computes standard evaluation metrics for regression.

    Parameters:
    - y_true (np.ndarray): True target values
    - y_pred (np.ndarray): Predicted target values

    Returns:
    - tuple: (mse, rmse, mae, r2)
    """
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)

    return mse, rmse, mae, r2
