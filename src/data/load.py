import pandas as pd
from typing import List

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def merge_datasets(datasets: List[pd.DataFrame]) -> pd.DataFrame:
    master_df = datasets[0]
    for i in range(1, len(datasets)):
        master_df = master_df.merge(datasets[i], how='inner')

    return master_df