#!/usr/bin/env python3
"""
Task 17
"""
from sklearn import model_selection


def split_data(df, target='Churn', test_size=0.2, random_state=42):
    """
    """
    X = df.drop(columns=[target])
    y = df[target]
    return model_selection.train_test_split(X, y,
                            test_size=test_size,
                            stratify=y,
                            random_state=random_state)
