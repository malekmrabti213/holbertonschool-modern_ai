#!/usr/bin/env python3
"""
Task 10
"""
import pandas as pd
import matplotlib.pyplot as plt


def plot_categorical_vs_churn(df, col):
    """
    """
    plt.figure(figsize=(12, 8))    
    # ct = pd.crosstab(df[col], df['Churn'], normalize='index')
    ct = df.groupby(col)['Churn'].value_counts(normalize=True).unstack()

    plt.bar(ct.index, ct['Yes'])
    plt.title(f'Churn Rate by {col}')
    plt.xticks(rotation=45)
    plt.ylabel('Churn Rate')
    plt.show()
