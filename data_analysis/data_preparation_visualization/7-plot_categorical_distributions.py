#!/usr/bin/env python3
"""
Task 7
"""
import matplotlib.pyplot as plt


def plot_categorical_distributions(df, columns_to_plot=None):
    """
    """
    if columns_to_plot is None:
        cat_cols = df.select_dtypes(include='object').columns.drop('Churn')
    else:
        cat_cols = [c for c in columns_to_plot if c in df.select_dtypes('object') and c != 'Churn']
    n_cols, n_rows = 3, (len(cat_cols)+2)//3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15,5*n_rows))
    axes = axes.flatten()
    for i, col in enumerate(cat_cols):
        counts = df[col].value_counts()
        axes[i].bar(counts.index, counts.values)
        axes[i].set_title(col)
        axes[i].tick_params(axis='x', rotation=45)
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.savefig("Task_7.png")
    plt.show()
