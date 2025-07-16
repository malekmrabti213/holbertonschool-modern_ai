#!/usr/bin/env python3
"""
Task 9
"""
import seaborn as sns
import matplotlib.pyplot as plt


def plot_correlation_heatmap(df):
    """
    """
    plt.figure(figsize=(6,5))

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    corr = df[numeric_cols].corr()

    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    plt.show()
