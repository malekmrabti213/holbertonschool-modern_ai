#!/usr/bin/env python3
"""
Task 3
"""


def train_model(model, X, Y, epochs, verbose=1):
    """
    Train a neural network model.

    Arguments:
    model -- compiled keras model
    X -- input data, shape (number of examples, input features)
    Y -- labels, shape (number of examples, 1)
    epochs -- number of training epochs
    verbose -- verbosity mode
    Returns:
    None
    """
    model.fit(X, Y, epochs=epochs, verbose=verbose)
