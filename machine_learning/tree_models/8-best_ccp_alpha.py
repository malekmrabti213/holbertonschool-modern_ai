#!/usr/bin/env python3
"""
Task 8
"""


def get_best_alpha(clfs, train_scores, test_scores, ccp_alphas):
    """
    Parameters:
        clfs (list): List of trained DecisionTreeClassifier instances.
        train_scores (list): Training accuracy scores for each model.
        test_scores (list): Test accuracy scores for each model.
        ccp_alphas (ndarray): ccp_alpha values corresponding to the models.

    Returns:
        best_alpha (float): Chosen ccp_alpha.
        best_clf (DecisionTreeClassifier): Corresponding best classifier.
    """
    max_test_acc = max(test_scores)

    # Indices where test accuracy is maximal
    best_indices = [i for i,
                     score in enumerate(test_scores) if score == max_test_acc]

    if len(best_indices) == 1:
        best_index = best_indices[0]
    else:
        # Compare train-test gap for all ties
        best_index = min(
            best_indices,
            key=lambda i: abs(train_scores[i] - test_scores[i])
        )

    best_alpha = ccp_alphas[best_index]
    best_clf = clfs[best_index]
    return best_alpha, best_clf
