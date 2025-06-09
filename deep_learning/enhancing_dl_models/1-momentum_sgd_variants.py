#!/usr/bin/env python3
"""
Task 1
"""

from tensorflow import keras


def get_optimizer_SGD(name, lr, momentum, nesterov=False):
    """
    Arguments:
    - name: A string indicating the optimizer variant. Choose from:
      - 'SGD': Standard stochastic gradient descent.
      - 'SGD+Momentum': SGD with classical momentum.
      - 'SGD+Momentum+Nesterov': SGD with momentum and Nesterov acceleration.

    - lr: A float specifying the learning rate for the optimizer.

    - momentum: A float specifying the momentum factor.

    - nesterov: A boolean indicating whether to apply Nesterov acceleration.

    Returns:
    - optimizer: A Keras SGD optimizer instance configured with the provided settings.
    """
    if name == 'SGD':
        return keras.optimizers.SGD(learning_rate=lr)
    if name == 'SGD+Momentum':
        return keras.optimizers.SGD(learning_rate=lr, momentum=momentum)
    if name == 'SGD+Momentum+Nesterov':
        return keras.optimizers.SGD(learning_rate=lr, momentum=momentum,
                                    nesterov=nesterov)
