from src.data.load import load_data, merge_datasets
from src.data.preprocess import preprocess
from src.config.load_config import load_config
from pathlib import Path

data_config = load_config('configs/data.yaml')

RAW_DIR = data_config['raw_data_dir']
PROCESSED_FILE = Path(data_config['processed_data_path'])

def main():
    build_dataset()
    df = load_data(PROCESSED_FILE)

def build_dataset():
    if PROCESSED_FILE.exists():
        return

    datasets = [load_data(f) for f in data_config['datasets']]
    merged_df = merge_datasets(datasets)
    preprocessed = preprocess(merged_df)

    preprocessed.to_csv(PROCESSED_FILE, index=False)


if __name__ == '__main__':
    main()