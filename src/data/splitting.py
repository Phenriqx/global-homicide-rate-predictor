from sklearn.model_selection import TimeSeriesSplit
import numpy as np

def split_data(X, y, test_size_ratio = 0.2, n_splits = 3):
    np.random.seed(42)
    tscv = TimeSeriesSplit(n_splits, test_size=round(test_size_ratio * len(X)))
    splits = list(tscv.split(X))

    return splits