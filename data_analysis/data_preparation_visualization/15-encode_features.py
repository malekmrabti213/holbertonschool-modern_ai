#!/usr/bin/env python3
"""
Task 15
"""
import pandas as pd
from sklearn import preprocessing


def encode_features(df):
    """
    """
    df = df.copy()
    binary_cols = ['Partner', 'Dependents',
                   'PaperlessBilling', 'SeniorCitizen']
    nominal_cols = ['Contract', 'PaymentMethod']

    le = preprocessing.LabelEncoder()
    binary_oe = preprocessing.OrdinalEncoder(categories=[['No', 'Yes']])
    tenure_oe = preprocessing.OrdinalEncoder()

    df['Churn'] = le.fit_transform(df['Churn'])

    for col in binary_cols:
        df[col] = binary_oe.fit_transform(df[[col]]).astype(int)

    df['TenureGroup'] = tenure_oe.fit_transform(df[['TenureGroup']])

    df = pd.get_dummies(df, columns=nominal_cols,
                        drop_first=True).astype(int)

    return df, le, binary_oe, tenure_oe
