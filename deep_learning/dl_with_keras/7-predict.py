#!/usr/bin/env python3
"""
Task 7
"""
import tensorflow as tf


def predict(model, X, verbose=0):
    """
    Make predictions using a trained Kera 
    model -- A trained Keras model.
    X -- Input data with a shape of (number of examples, input features).
    verbose -- Verbosity mode for prediction (default is 0).

    Returns:
    predictions -- A list of predicted class labels.
    """
    probabilities = model.predict(X, verbose=verbose)
    predictions = tf.argmax(probabilities, axis=1)
    return predictions
