#!/usr/bin/env python3
"""
Task 6
"""


def save_model_weights(model, filepath):
    """
    Save only the weights of a Keras model.

    Arguments:
    model -- A trained Keras model whose weights need to be saved.
    filepath -- where the weights will be saved.
    """
    model.save_weights(filepath)


def load_model_weights(model, filepath):
    """
    Load weights into a Keras model from a specified filepath.

    Arguments:
    model -- A compatible Keras model where the weights will be loaded.
    filepath -- from where the weights will be loaded.

    Returns:
    None. The weights are loaded into the provided model.
    """
    model.load_weights(filepath)
