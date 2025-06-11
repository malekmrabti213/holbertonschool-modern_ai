#!/usr/bin/env python3
"""
Task 9
"""

from keras_tuner import Hyperband, RandomSearch, BayesianOptimization


def initiate_tuner(tuner_type, build_model, seed, hyperband_iterations,
                   max_trials, objective, overwrite=True):

    """
    Parameters:
    - tuner_type (str): Type of tuner to use
    - build_model (function): Function that returns a compiled Keras model
    - seed (int): Maximum number of epochs for each trial.
    - hyperband_iterations (int): Number of iterations for only Hyperband
    - max_trials (int): Maximum number 'RandomSearch','BayesianOptimization'
    - objective (str): Metric to optimize
    - overwrite (bool): Whether to overwrite the previous tuning project
    Returns:
    - tuner (kerastuner.Tuner): Configured Keras Tuner object.
    """

    if tuner_type == "Hyperband":

        tuner = Hyperband(
            build_model,
            objective=objective,
            seed=seed,
            hyperband_iterations=hyperband_iterations,
            overwrite=overwrite
        )

    elif tuner_type == "RandomSearch":

        tuner = RandomSearch(
            build_model,
            objective=objective,
            seed=seed,
            max_trials=max_trials,
            overwrite=overwrite
        )

    elif tuner_type == "BayesianOptimization":

        tuner = BayesianOptimization(
            build_model,
            objective=objective,
            seed=seed,
            max_trials=max_trials,
            overwrite=overwrite
        )

    return tuner
