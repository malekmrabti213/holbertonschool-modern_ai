#!/usr/bin/env python3
"""
Task 10
"""


def search_and_return_best_model(tuner, x_train, y_train,
                                 epochs, validation_split, verbose=0):
    """
    Parameters:
    - tuner (kerastuner.Tuner): A Keras Tuner instance
    - x_train (ndarray): Training input data.
    - y_train (ndarray): Training target data.
    - epochs (int): Number of epochs to train each model during tuning.
    - validation_split (float): Fraction of training data for validation.

    Returns:
    - best_hyperparameters (kerastuner.HyperParameters)
    """
    tuner.search(x_train, y_train, epochs=epochs,
                 validation_split=validation_split, verbose=verbose)
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

    return best_hyperparameters
