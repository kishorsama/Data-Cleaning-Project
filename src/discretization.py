import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

def equal_width(df:pd.DataFrame, column_name: str,bin_number:int=2) -> pd.DataFrame:
    df_clean = df.copy()
    
    if column_name in df.columns:
        df[column_name] = pd.cut(df[column_name], bins=bin_number, labels=False)
    return df_clean

def equal_frequency(df:pd.DataFrame, column_name:str, quartile_number:int=4):
    df_clean = df.copy()

    if column_name in df.columns:
        df[column_name] = pd.qcut(df[column_name], q=quartile_number)

    return df_clean