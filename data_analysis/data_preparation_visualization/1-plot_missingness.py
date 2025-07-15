#!/usr/bin/env python3
"""
Task 1
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_missingness(df):
    """
    """
    plt.figure(figsize=(12, 8))
    # Create a grid of positions where values are missing
    missing = df.isnull()

    # Create scatter plot positions
    row_indices, col_indices = np.where(missing)

    # Plot missing values
    plt.scatter(row_indices, col_indices, marker='|')

    # Set y-axis to show column names
    plt.yticks(range(len(df.columns)), df.columns)

    plt.title('Missingness Plot')
    plt.tight_layout()
    plt.show()
