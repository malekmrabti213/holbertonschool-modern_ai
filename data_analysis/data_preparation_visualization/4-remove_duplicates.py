#!/usr/bin/env python3
"""
Task 4
"""


def remove_duplicates(df):
    """
    """
    df = df.copy()
    return df.drop_duplicates()
