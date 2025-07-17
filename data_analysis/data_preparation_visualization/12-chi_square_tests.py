#!/usr/bin/env python3
"""
Task 12
"""
import pandas as pd
from scipy import stats


def chi_square_tests(df):
    """
    """
    results = {}
    for col in df.select_dtypes(include='object').columns.drop('Churn'):
        ct = pd.crosstab(df[col], df['Churn'])
        _, p, _, _ = stats.chi2_contingency(ct)
        results[col] = p
    return results
