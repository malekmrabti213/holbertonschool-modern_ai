#!/usr/bin/env python3
"""
Task 7
"""

from tensorflow import keras


def get_early_stopping_callback(patience, monitor='val_loss',
                                verbose=1):
    """
    Parameters:
    - patience (int): Number of epochs before stopping.
    - monitor (str): Metric to monitor, e.g., 'val_loss' or 'val_accuracy'.
    - verbose (int): Verbosity mode
    Returns:
    - keras.callbacks.EarlyStopping: Configured EarlyStopping
    """
    return keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience,
        restore_best_weights=True,
        verbose=verbose
    )
