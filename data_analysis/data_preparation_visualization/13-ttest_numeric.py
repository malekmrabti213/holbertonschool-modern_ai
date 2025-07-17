#!/usr/bin/env python3
"""
Task 13
"""
from scipy import stats


def ttest_numeric(df):
    """
    """
    results = {}
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    for col in numeric_cols:
        stat, p = stats.ttest_ind(
            df[df['Churn']=='Yes'][col],
            df[df['Churn']=='No'][col],
            equal_var=False
        )
        results[col] = p
    return results
