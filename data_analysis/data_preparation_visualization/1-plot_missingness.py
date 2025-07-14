#!/usr/bin/env python3
"""
Task 1
"""
import matplotlib.pyplot as plt
import seaborn as sns


def plot_missingness(df):
    """
    """
    plt.figure(figsize=(8,4))
    plt.imshow(df.isnull().T, aspect='auto')
    plt.yticks(range(len(df.columns)), df.columns)
    plt.title('Missingness Heatmap')
    plt.show()
