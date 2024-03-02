import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
class OutlierReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)  
        self.upper_bounds = {}
        self.lower_bounds = {}
        for column in X.columns:
            q1 = X[column].quantile(0.25)
            q3 = X[column].quantile(0.75)
            iqr = q3 - q1
            self.upper_bounds[column] = q3 + self.factor * iqr
            self.lower_bounds[column] = q1 - self.factor * iqr
        return self

    def transform(self, X, y=None):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)  
        X_transformed = X.copy()
        for column in X_transformed.columns:
            X_transformed[column] = np.where(X_transformed[column] > self.upper_bounds[column], self.upper_bounds[column], X_transformed[column])
            X_transformed[column] = np.where(X_transformed[column] < self.lower_bounds[column], self.lower_bounds[column], X_transformed[column])
        return X_transformed
