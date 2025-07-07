#!/usr/bin/env python3
"""
Task 6
"""
from sklearn import svm


def get_SVM_model(name, random_state):
    """
    Returns an untrained SVM classifier based on the specified kernel.

    Parameters:
    - name (str): One of 'linear', 'poly', 'rbf' indicating the kernel type.
    - random_state (int): Seed for reproducibility.

    Returns:
    - SVC instance configured with the specified kernel and random_state.
    """
    if name == 'linear':
        model = svm.SVC(kernel='linear', random_state=random_state)
    elif name == 'poly':
        model = svm.SVC(kernel='poly', random_state=random_state)
    elif name == 'rbf':
        model = svm.SVC(kernel='rbf', random_state=random_state)
    else:
        raise ValueError(f"Invalid kernel name '{name}'.")

    return model
