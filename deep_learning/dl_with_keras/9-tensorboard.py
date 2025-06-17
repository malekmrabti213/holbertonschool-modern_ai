#!/usr/bin/env python3
"""
Task 9
"""
from tensorflow import keras
import datetime


def log_to_tensorboard(log_dir, model, X, Y, epochs, verbose=1):
    """
    Train a neural network and log training progress in TensorBoard.

    Arguments:
    log_dir -- Directory to save TensorBoard logs.
    model -- compiled keras model
    X -- input data, shape (number of examples, input features)
    Y -- labels, shape (number of examples, 1)
    epochs -- number of training epochs
    verbose -- verbosity mode.

    Returns:
    None
    """

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir=f"{log_dir}/{timestamp}",
    histogram_freq=1)

    model.fit(
        X, Y,
        epochs=epochs,
        callbacks=[tensorboard_callback], 
        verbose=verbose
    )
