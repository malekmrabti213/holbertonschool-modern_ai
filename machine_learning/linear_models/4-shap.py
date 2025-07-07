#!/usr/bin/env python3
"""
Task 4
"""
import shap


def get_shap_explainer_and_values(model, X_train, X_test):
    """
    Creates a SHAP explainer and computes SHAP values.

    Parameters:
        model: Trained model
        X_train: Training data (used as background)
        X_test: Data to explain

    Returns:
        explainer: SHAP explainer instance
        shap_values: SHAP values for X_test
    """
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    return explainer, shap_values
