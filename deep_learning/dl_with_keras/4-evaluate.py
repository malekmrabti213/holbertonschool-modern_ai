#!/usr/bin/env python3
"""
Task 4
"""


def evaluate_model(model, X, Y, verbose=0):
    """
    Evaluate a trained neural network model.

    Arguments:
    model -- trained keras model
    X -- input data, shape (number of examples, input features)
    Y -- true labels, shape (number of examples, 1)

    Returns:
    loss -- loss on the data
    accuracy -- accuracy on the data
    """
    loss, accuracy = model.evaluate(X, Y, verbose=0)
    return loss, accuracy
