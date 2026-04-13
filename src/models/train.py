from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from src.models.pipeline import build_preprocessor
from src.data.splitting import split_data
from src.config.load_config import load_config

model_config = load_config('configs/model.yaml')

RF_HYPERPARAMETERS = model_config['random_forest']

def train_model(X, y, num_features, cat_features):
    preprocessor = build_preprocessor(num_features, cat_features)

    model_pipeline = Pipeline([
        ('preprocessor' ,preprocessor),
        ('model', RandomForestRegressor(
            n_estimators=RF_HYPERPARAMETERS['n_estimators'],
            max_features=RF_HYPERPARAMETERS['max_features']
        ))
    ])

    splits = split_data(X, y)
    for i, (train_index, test_index) in enumerate(splits):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model_pipeline.fit(X_train, y_train)

    return model_pipeline