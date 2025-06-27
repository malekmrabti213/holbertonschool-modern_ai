#!/usr/bin/env python3
"""
Task 7
"""
from sklearn import tree


def prune_and_evaluate_trees(X_train, y_train, X_test,
                             y_test, ccp_alphas,
                             random_state, min_samples_leaf,
                             min_samples_split):
    """
    Parameters:
        X_train, y_train: Training data and labels
        X_test, y_test: Testing data and labels
        ccp_alphas (ndarray): Array of ccp_alpha values
        random_state (int): Seed for reproducibility
        min_samples_leaf(int):Minimum samples per leaf
        min_samples_split(int):Minimum samples required to split a node

    Returns:
        clfs (list): List of trained DecisionTreeClassifier instances
        train_scores (list): Training accuracy scores for each model
        test_scores (list): Testing accuracy scores for each model
    """
    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = tree.DecisionTreeClassifier(ccp_alpha=ccp_alpha,
                                          random_state=random_state,
                                          min_samples_leaf=min_samples_leaf,
                                          min_samples_split=min_samples_split)
        clf.fit(X_train, y_train)
        clfs.append(clf)

    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]

    return clfs, train_scores, test_scores
