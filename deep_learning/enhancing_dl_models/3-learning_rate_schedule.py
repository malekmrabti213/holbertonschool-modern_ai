#!/usr/bin/env python3
"""
Task 3
"""

from tensorflow import keras


def get_optimizer_SGD_with_schedule(schedule_type, initial_lr,
                                    decay_steps, decay_rate, momentum):
    """
    Parameters:
    - schedule_type (str): Type of learning rate schedule to use.
        - 'exponential': Uses ExponentialDecay.
        - 'inverse_time': Uses InverseTimeDecay.
    - initial_lr (float): Initial learning rate before decay is applied.
    - decay_steps (int): Number of steps before applying decay.
    - decay_rate (float): The decay rate factor.
    - momentum (float): Momentum value for the SGD optimizer.

    Returns:
    - optimizer (tf.keras.optimizers.SGD)
    - lr_schedule (tf.keras.optimizers.schedules.LearningRateSchedule)

    """
    if schedule_type == 'exponential':
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=True
        )
    elif schedule_type == 'inverse_time':
        lr_schedule = keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=True
        )
    return keras.optimizers.SGD(learning_rate=lr_schedule,
                                momentum=momentum), lr_schedule
