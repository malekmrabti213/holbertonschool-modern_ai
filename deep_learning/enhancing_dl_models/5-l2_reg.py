#!/usr/bin/env python3
"""
Task 5
"""

from tensorflow import keras


def build_model_with_L2_regularization(input_dim, hidden_units,
                                       n_layers, lambda_l2):
    """
    Builds a neural network model with L2 regularization.

    Parameters:
    - input_dim (int): The number of input features.
    - hidden_units (int): The number of units in each hidden layer.
    - n_layers (int): The number of hidden layers in the model.
    - lambda_l2 (float): The L2 regularization strength.

    Returns:
    - model: A Keras model instance (not compiled).
    """
    reg = keras.regularizers.l2(lambda_l2)
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=(input_dim,)))
    for _ in range(n_layers):
        model.add(keras.layers.Dense(hidden_units, activation='relu',
                                     kernel_regularizer=reg))
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model
