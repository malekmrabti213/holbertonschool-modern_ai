#!/usr/bin/env python3
"""
Task 2
"""
import pandas as pd


def convert_total_charges(df):
    """
    """
    df = df.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    return df
