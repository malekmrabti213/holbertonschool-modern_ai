#!/usr/bin/env python3
"""
Task 1
"""
from tensorflow import keras


def build_model(input_dim, neurons_h):
    """
    Arguments:
    input_dim -- number of input features

    Returns:
    model -- keras model
    """
    input_layer = keras.layers.Input(shape=(input_dim,))
    hidden_layer = keras.layers.Dense(neurons_h, activation='sigmoid')(input_layer)
    output_layer = keras.layers.Dense(10, activation='softmax')(hidden_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model