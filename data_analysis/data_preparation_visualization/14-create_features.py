#!/usr/bin/env python3
"""
Task 14
"""
import pandas as pd


def create_features(df):
    """
    """
    df = df.copy()

    services = [
        'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    service_counts = df[services].eq('Yes').astype(int)

    internet_subscribed = (
        df['InternetService']
        .isin(['DSL', 'Fiber optic'])
        .astype(int)
    )
    df['NumServices'] = service_counts.sum(axis=1) + internet_subscribed

    df['TenureGroup'] = (
        pd.cut(df['tenure'],
               bins=[0, 12, 24, 48, 60, float('inf')],
               labels=['0-12', '13-24', '25-48', '49-60', '60+']))

    df.drop(columns=['tenure', 'MultipleLines', 'OnlineSecurity',
                     'OnlineBackup', 'DeviceProtection',
                     'TechSupport', 'StreamingTV',
                     'StreamingMovies', 'InternetService'], inplace=True)
    return df
