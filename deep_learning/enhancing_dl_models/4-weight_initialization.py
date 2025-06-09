#!/usr/bin/env python3
"""
Task 4
"""

from tensorflow import keras


def build_model_initializer_by_activation(input_dim, hidden_units,
                                          activation):
    """
    Parameters:
    - input_dim (int): The number of input features.
    - hidden_units (int): The number of neurons in the hidden layer.
    - activation (str): The activation function to use in the hidden layer.
        - 'sigmoid' or 'tanh': Uses GlorotUniform initializer.
        - 'relu' or 'leaky_relu': Uses HeNormal initializer.

    Returns:
    - model: A Keras model instance (not compiled).
    """

    if activation == 'sigmoid' or activation == 'tanh':
        initializer = keras.initializers.GlorotUniform()
    elif activation == 'relu' or activation == 'leaky_relu':
        initializer = keras.initializers.HeNormal()

    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=(input_dim,)))

    if activation == 'leaky_relu':
        model.add(keras.layers.Dense(hidden_units,
                                     kernel_initializer=initializer))
        model.add(keras.layers.LeakyReLU())
    else:
        model.add(keras.layers.Dense(hidden_units, activation=activation,
                                     kernel_initializer=initializer))

    model.add(keras.layers.Dense(10, activation='softmax'))
    return model
