from src.data.load import load_data, merge_datasets
from src.data.preprocess import preprocess
from src.config.load_config import load_config
from src.data.splitting import build_features
from src.models.train import train_or_load_model
from src.models.evaluate import evaluate_model

from pathlib import Path

data_config = load_config('configs/data.yaml')

RAW_DIR = data_config['raw_data_dir']
PROCESSED_FILE = Path(data_config['processed_data_path'])

def main():
    build_dataset()
    df = load_data(PROCESSED_FILE)

    df, X, y = build_features(df)
    num_features = ['hdi', 'gdp-per-capita', 'gini-coefficient', 'urbanization', 'rule-of-law-index', 'political-corruption-index']
    cat_features = ['entity', 'world-region']

    model_pipeline = train_or_load_model(X, y, num_features, cat_features)
    mae, rmse, r2 = evaluate_model(model_pipeline, X, y)

    print("Evaluating Model...")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"R2: {r2}")

def build_dataset():
    if PROCESSED_FILE.exists():
        return

    datasets = [load_data(f) for f in data_config['datasets']]
    merged_df = merge_datasets(datasets)
    preprocessed = preprocess(merged_df)

    preprocessed.to_csv(PROCESSED_FILE, index=False)


if __name__ == '__main__':
    main()