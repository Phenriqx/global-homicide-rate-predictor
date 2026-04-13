from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd

def split_data(X, y, test_size_ratio = 0.2, n_splits = 3):
    np.random.seed(42)
    tscv = TimeSeriesSplit(n_splits, test_size=round(test_size_ratio * len(X)))
    splits = []

    for train_index, test_index in tscv.split(X):
        splits.append((
            X.iloc[train_index],
            X.iloc[test_index],
            y.iloc[train_index],
            y.iloc[test_index],
        ))

    return splits

def build_features(df: pd.DataFrame):
    df = df.sort_values('year').reset_index(drop=True)
    features = df.drop(['code', 'homicide-rate-100000'], axis=1)
    X = features
    y = df['homicide-rate-100000']

    return df, X, y