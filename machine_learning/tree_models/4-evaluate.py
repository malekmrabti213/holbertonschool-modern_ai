#!/usr/bin/env python3
"""
Task 4
"""
from sklearn import metrics


def evaluate(true_labels, predicted_labels, class_names):
    """
    Generates a classification report comparing true and predicted labels.

    Args:
        true_labels: Ground truth labels (1D array-like)
        predicted_labels: Predicted labels (1D array-like)
        class_names: List of class names corresponding to label indices

    Returns:
        A string containing the classification report
    """
    report = metrics.classification_report(true_labels,
                                           predicted_labels,
                                           target_names=class_names)
    return report
