#!/usr/bin/env python3
"""
Task 3
"""
from sklearn import linear_model


def lasso_regression(random_state):
  
  return linear_model.Lasso(random_state=random_state)
