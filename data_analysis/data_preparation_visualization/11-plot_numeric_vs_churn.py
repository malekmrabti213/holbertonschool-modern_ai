#!/usr/bin/env python3
"""
Task 11
"""
import matplotlib.pyplot as plt


def plot_numeric_vs_churn(df, col):
    """
    """
    plt.figure(figsize=(12, 8))
    yes = df[df['Churn'] == 'Yes'][col].dropna()
    no  = df[df['Churn'] == 'No'][col].dropna()
    plt.hist([no, yes], bins = 30, label=['No', 'Yes'])
    plt.title(f'{col} Distribution by Churn')
    plt.xlabel(col)
    plt.legend(title='Churn')
    plt.show()
