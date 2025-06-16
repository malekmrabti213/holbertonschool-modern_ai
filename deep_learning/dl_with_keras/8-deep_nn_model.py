#!/usr/bin/env python3
"""
Task 8
"""
from tensorflow import keras


def build_deep_model(input_dim, hidden_layers):
    """
    Build a deep neural network model using the Sequential API.

    Arguments:
    input_dim -- number of input features
    hidden_layers -- list of number of neurons in a hidden layer

    Returns:
    model -- Keras model
    """
    model = keras.models.Sequential()

    model.add(keras.layers.Input(shape=(input_dim,)))

    for neurons in hidden_layers:
        model.add(keras.layers.Dense(neurons, activation='relu'))

    model.add(keras.layers.Dense(10, activation='softmax'))

    return model
