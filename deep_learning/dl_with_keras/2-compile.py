#!/usr/bin/env python3
"""
Task 2
"""
from tensorflow import keras


def compile_model(model, learning_rate=0.01):
    """
    Arguments:
    model -- keras model
    learning_rate -- learning rate for SGD optimizer (default is 0.01)
    """
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
