#!/usr/bin/env python3
"""
Task 2
"""

from tensorflow import keras

def get_optimizer(name, learning_rate, momentum, beta_1, beta_2, rho):
    """
    Returns a Keras optimizer based on the given name.

    Parameters:
    - name (str): Optimizer name — 'sgd', 'adam', or 'rmsprop'.
    - learning_rate (float): Learning rate for the optimizer.
    - momentum (float): Momentum value (only used for SGD).
    - beta_1 (float): Exponential decay rate for the 1st moment estimate (only used for Adam).
    - beta_2 (float): Exponential decay rate for the 2nd moment estimate (only used for Adam).
    - rho (float): Decay factor for RMSProp (only used for RMSProp).

    Returns:
    - optimizer: A tf.keras.optimizers.Optimizer instance.
    """
    if name == 'sgd':
        return keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    if name == 'adam':
        return keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
    if name == 'rmsprop':
        return keras.optimizers.RMSprop(learning_rate=learning_rate, rho=rho)
