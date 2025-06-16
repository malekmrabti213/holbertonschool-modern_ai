#!/usr/bin/env python3
"""
Task 5
"""
from tensorflow import keras


def save_model(model, filepath):
    """
    Save a Keras model including architecture, weights, and optimizer state.

    Arguments:
    model -- A trained Keras model to be saved.
    filepath -- where the model will be saved.
    """
    model.save(filepath)

def load_model(filepath):
    """
    Load a saved Keras model from a specified filepath.

    Arguments:
    filepath -- from where the model will be loaded.

    Returns:
    model -- The reloaded Keras model.
    """
    model = keras.models.load_model(filepath)
    return model
