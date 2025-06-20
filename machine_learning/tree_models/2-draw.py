#!/usr/bin/env python3
"""
Task 2
"""
from sklearn import tree


def draw(clf, feature_names,class_names):
    """
    Displays the textual representation of a trained Decision Tree.

    Parameters:
        clf (DecisionTreeClassifier): Trained decision tree classifier.
        feature_names (list): feature names.
        class_names (list): target classes  
    """
    tree_text = tree.export_text(clf, feature_names=feature_names,
                                 class_names=class_names)
    print(tree_text)
