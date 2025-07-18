#!/usr/bin/env python3
"""
Task 14
"""
import pandas as pd


def create_features(df):
    """
    """
    df = df.copy()

    # Define the service columns to consider
    services = [
        'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]

    # Create a binary mask for each service column
    service_counts = df[services].eq('Yes').astype(int)

    # Include InternetService: count 'DSL' and 'Fiber optic' as subscribed
    internet_subscribed = (
        df['InternetService']
        .isin(['DSL', 'Fiber optic'])
        .astype(int)
    )

    # Sum all services to compute NumServices
    df['NumServices'] = service_counts.sum(axis=1) + internet_subscribed

    # Bin the tenure into defined groups
    df['TenureGroup'] = (
        pd.cut(df['tenure'],
               bins=[0, 12, 24, 48, 60, float('inf')],
               labels=['0-12', '13-24', '25-48', '49-60', '60+']))

    # Drop the original 'tenure' column
    df.drop(columns=['tenure','MultipleLines', 'OnlineSecurity',
                     'OnlineBackup', 'DeviceProtection',
                     'TechSupport', 'StreamingTV',
                     'StreamingMovies', 'InternetService'], inplace=True) 
    return df
