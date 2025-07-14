#!/usr/bin/env python3
"""
Task 5
"""


def drop_customerID(df):
    """
    """
    df = df.copy()
    return df.drop(columns=['customerID'])
