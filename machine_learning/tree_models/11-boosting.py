#!/usr/bin/env python3
"""
Task 11
"""
from sklearn import ensemble
import xgboost as xgb
import lightgbm as lgb


def compare_boosting_classifiers(name, n_estimators, random_state):
    """
    Returns a boosting model based on the specified name.

    Parameters:
    - name (str): The name of the model to return. Choose from:
        - 'adaboost': AdaBoostClassifier
        - 'gradientboosting': GradientBoostingClassifier
        - 'xgboost': XGBClassifier
        - 'lightgbm': LGBMClassifier
    - n_estimators (int): Number of estimators (trees).
    - random_state (int): Seed for reproducibility.

    Returns:
    - model: An untrained boosting classifier
    """
    if name == 'adaboost':
        return ensemble.AdaBoostClassifier(n_estimators=n_estimators,
                                  random_state=random_state)
    elif name == 'gradientboosting':
        return ensemble.GradientBoostingClassifier(n_estimators=n_estimators,
                                          random_state=random_state)
    elif name == 'xgboost':
        return xgb.XGBClassifier(n_estimators=n_estimators,
                                 random_state=random_state)
    elif name == 'lightgbm':
        return lgb.LGBMClassifier(n_estimators=n_estimators,
                                  random_state=random_state, verbose=-1)
    else:
        raise ValueError(f"Unknown model name '{name}'")
