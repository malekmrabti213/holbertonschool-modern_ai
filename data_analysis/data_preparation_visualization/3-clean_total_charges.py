#!/usr/bin/env python3
"""
Task 3
"""


def clean_total_charges(df, method='drop'):
    """
    """
    df = df.copy()
    if method == 'drop':
        df = df.dropna(subset=['TotalCharges'])
    elif method == 'median':
        df.fillna({'TotalCharges': df['TotalCharges'].median()}, inplace=True)
    else:
        mask = df['TotalCharges'].isna()
        df.loc[mask, 'TotalCharges'] = (
            df.loc[mask, 'MonthlyCharges'] * df.loc[mask, 'tenure'])
    return df
