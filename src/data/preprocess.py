import pandas as pd

def to_lower_column(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [col.lower() for col in df.columns]
    return df

def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop('gdp-per-capita-(annotations)', axis=1)

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        'Human Development Index': 'hdi',
        'World region according to OWID': 'world-region',
        'Homicide rate per 100,000 population': 'homicide-rate-100000',
        'urban-population-(%-of-total-population)': 'urbanization',
    }, axis=1)

    df = to_lower_column(df)
    df.columns = df.columns.str.replace(' ', '-')
    df = drop_unused_columns(df)