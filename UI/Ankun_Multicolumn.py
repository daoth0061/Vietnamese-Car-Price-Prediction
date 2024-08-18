from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns  # array of column names to encode
        self.encoders = {col: LabelEncoder() for col in columns}
    
    def fit(self, X, y=None):
        for col in self.columns:
            self.encoders[col].fit(X[col])
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for col in self.columns:
            X_copy[col] = self.encoders[col].transform(X_copy[col])
        return X_copy
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, selected_columns):
        self.selected_columns = selected_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.selected_columns]