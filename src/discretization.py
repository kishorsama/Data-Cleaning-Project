import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

def equal_width(df:pd.DataFrame, column_name: str, bin_number:int=2) -> pd.DataFrame:
    df_clean = df.copy()
    
    if column_name in df.columns:
        df_clean[column_name] = pd.cut(df_clean[column_name], bins=bin_number, labels=False)
    
    return df_clean


def equal_frequency(df:pd.DataFrame, column_name:str, quartile_number:int=4):
    df_clean = df.copy()

    if column_name in df.columns:
        df_clean[column_name] = pd.qcut(df_clean[column_name], q=quartile_number, duplicates='drop', labels=False)

    return df_clean

def kbins(df:pd.DataFrame, column_name:list, bin_number: int=5, strat:str='kmeans') -> pd.DataFrame:
    df_clean = df.copy()

    kbd = KBinsDiscretizer(encode='ordinal', strategy= strat, n_bins= bin_number)

    df_clean[column_name] = kbd.fit_transform(df_clean[column_name])

    return df_clean 