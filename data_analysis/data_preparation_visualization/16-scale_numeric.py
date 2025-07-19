#!/usr/bin/env python3
"""
Task 16
"""
from sklearn import preprocessing


def scale_numeric(df):
    """
    """
    df = df.copy()
    scaler = preprocessing.StandardScaler()
    num_cols = ['MonthlyCharges','TotalCharges']
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df
