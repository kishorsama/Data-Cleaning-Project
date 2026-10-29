import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def min_max_scaling(df: pd.DataFrame, column_name:list) -> pd.DataFrame:

    df_clean = df.copy()
    mmscaler = MinMaxScaler()

    df_clean[column_name] = mmscaler.fit_transform(df_clean[column_name])
    
    return df_clean


def standard_scaling(df: pd.DataFrame, column_name:list) -> pd.DataFrame:

    df_clean = df.copy()
    scaler = StandardScaler()

    df_clean[column_name] = scaler.fit_transform(df_clean[column_name])
    
    return df_clean