#!/usr/bin/env python3
"""
Task 3
"""


def generate_predictions(clf, X):
    """
    Generates predictions using a trained classifier.

    Args:
        clf: A trained Scikit-learn classifier instance.
        X: Feature matrix (NumPy array or pandas DataFrame).

    Returns:
        A NumPy array containing the predicted class labels.
    """
    return clf.predict(X)
