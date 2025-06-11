#!/usr/bin/env python3
"""
Task 8
"""

from tensorflow import keras


def build_model(hp):
    """
    Parameters:
    - hp (kerastuner.HyperParameters)


    Returns:
    - keras.Sequential: A compiled Keras Sequential model
    """
    model = keras.Sequential()

    model.add(keras.layers.Input(shape=(784,)))

    for _ in range(hp.Int('num_layers', 1, 2)):
        model.add(keras.layers.Dense(
            units=hp.Int('units', min_value=4, max_value=12, step=4),
            activation=hp.Choice('activation', ['relu', 'sigmoid']),
        ))

    model.add(keras.layers.Dense(10, activation='softmax'))

    lr = hp.Choice('learning_rate', values=[1e-2, 1e-3])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
