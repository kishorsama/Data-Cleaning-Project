import pandas as pd

def drop(df: pd.DataFrame, axix: int) ->pd.DataFrame:
    df_clean = df.copy()
    print(f"total null values: {df_clean.isna().sum()}")

    #column
    if axix == 1: 
        df_clean = df_clean.dropna(axis=1)
    #rows
    else:
        df_clean = df_clean.dropna(axis=0)

    print('Done drop!')
    return df_clean


def forward_fill(df: pd.DataFrame, axix: int) ->pd.DataFrame:
    df_clean = df.copy()
    print(f"total null values: {df_clean.isna().sum()}")

    if axix == 1: 
        df_clean = df_clean.ffill(axis=1)
        df_clean = df_clean.bfill(axis=1)
    #rows
    else:
        df_clean = df_clean.ffill(axis=0)
        df_clean = df_clean.bfill(axis=0)

    print('Done forward fill!')
    return df_clean


def backward_fill(df: pd.DataFrame, axix: int) ->pd.DataFrame:
    df_clean = df.copy()
    print(f"total null values: {df_clean.isna().sum()}")

    if axix == 1: 
        df_clean = df_clean.bfill(axis=1)
        df_clean = df_clean.ffill(axis=1)
        
    #rows
    else:
        df_clean = df_clean.bfill(axis=0)
        df_clean = df_clean.ffill(axis=0)
        

    print('Done backward fill!')
    return df_clean
