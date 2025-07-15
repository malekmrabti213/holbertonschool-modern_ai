#!/usr/bin/env python3
"""
Task 2
"""
import pandas as pd


def convert_columns(df):
    """
    """
    df = df.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['SeniorCitizen'] = df['SeniorCitizen'].astype('object')
    return df
