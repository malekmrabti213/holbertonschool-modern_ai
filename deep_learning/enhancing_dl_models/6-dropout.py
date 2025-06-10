#!/usr/bin/env python3
"""
Task 6
"""

from tensorflow import keras


def build_model_with_dropout(input_dim, hidden_units, n_layers,
                             dropout_rate_input, dropout_rate_hidden):
    """
    Parameters:
    - input_dim (int): The number of input features.
    - hidden_units (int): The number of units in each hidden layer.
    - n_layers (int): The number of hidden layers in the model.
    - dropout_rate_input (float): The dropout rate:input layer.
    - dropout_rate_hidden (float): The dropout rate:hidden layer.

    Returns:
    - model: A Keras model instance (not compiled).
    """
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=(input_dim,)))
    model.add(keras.layers.Dropout(dropout_rate_input))
    for _ in range(n_layers):
        model.add(keras.layers.Dense(hidden_units, activation='relu'))
        model.add(keras.layers.Dropout(dropout_rate_hidden))
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model
