#!/usr/bin/env python3
"""
Task 0
"""
from tensorflow import keras


def build_model(input_dim, neurons_h):
    """
    Arguments:
    input_dim -- number of input features

    Returns:
    model -- keras model
    """
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=(input_dim,)))
    model.add(keras.layers.Dense(neurons_h, activation='sigmoid'))
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model
