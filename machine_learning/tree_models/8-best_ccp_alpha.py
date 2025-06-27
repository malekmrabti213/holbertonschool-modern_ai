#!/usr/bin/env python3
"""
Task 8
"""


def get_best_alpha(clfs, train_scores, test_scores, ccp_alphas):
    """
    Selects the best ccp_alpha based on test accuracy and generalization gap.

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

    # Find indices with the maximum test accuracy
    best_indices = [i for i,
                    score in enumerate(test_scores) if score == max_test_acc]

    if len(best_indices) == 1:
        best_index = best_indices[0]
    else:
        # Compute generalization gap for each candidate
        gaps = [abs(train_scores[i] - test_scores[i]) for i in best_indices]
        min_gap = min(gaps)

        # Indices with smallest generalization gap
        min_gap_indices = [i for i,
                           g in zip(best_indices, gaps) if g == min_gap]

        if len(min_gap_indices) == 1:
            best_index = min_gap_indices[0]
        else:
            # pick the one with largest ccp_alpha (simpler model)
            best_index = max(min_gap_indices, key=lambda i: ccp_alphas[i])

    best_alpha = ccp_alphas[best_index]
    best_clf = clfs[best_index]
    return best_alpha, best_clf
