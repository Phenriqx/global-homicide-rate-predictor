from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

import pandas as pd
import numpy as np
from src.data.splitting import split_data

def evaluate_model(model: Pipeline, X: pd.DataFrame, y: pd.DataFrame):
    splits = split_data(X, y)
    fold_scores = []

    for i, (X_train, X_test, y_train, y_test) in enumerate(splits):
        preds = model.predict(X_test)
        fold_scores.append({
            'mae': mean_absolute_error(y_test, preds),
            'rmse': np.sqrt(mean_squared_error(y_test, preds)),
            'r2': r2_score(y_test, preds)
        })

    scores = pd.DataFrame(fold_scores)
    mae = scores['mae'].mean()
    rmse = scores['rmse'].mean()
    r2 = scores['r2'].mean()

    return mae, rmse, r2