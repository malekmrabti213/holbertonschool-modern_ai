#!/usr/bin/env python3
"""
Task 10
"""
import numpy as np


def feature_importance(rf):
    """
    Arguments:
    - rf: Trained RandomForestClassifier

    Returns:
    - importances: Array of feature importance scores
    - indices: Indices of features sorted from most to least important
    """
    importances = rf.feature_importances_
    indices = np.argsort(importances)

    return importances, indices
