import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PowerTransformer

from src.models.pipeline import build_preprocessor
from src.data.splitting import split_data
from src.config.load_config import load_config

from pathlib import Path

model_config = load_config('configs/model.yaml')

MODEL_PATH = Path(model_config['model_path'])
RF_HYPERPARAMETERS = model_config['random_forest']

def train_or_load_model(X, y, num_features, cat_features):
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        return model

    preprocessor = build_preprocessor(num_features, cat_features)

    model_pipeline = Pipeline([
        ('preprocessor' ,preprocessor),
        ('model', TransformedTargetRegressor(regressor=RandomForestRegressor(
            n_estimators=RF_HYPERPARAMETERS['n_estimators'],
            max_features=RF_HYPERPARAMETERS['max_features']
        ), transformer=PowerTransformer(method='yeo-johnson')))
    ])

    splits = split_data(X, y)
    for i, (X_train, X_test, y_train, y_test) in enumerate(splits):
        model_pipeline.fit(X_train, y_train)

    joblib.dump(model_pipeline, 'models/random_forest.joblib')

    return model_pipeline