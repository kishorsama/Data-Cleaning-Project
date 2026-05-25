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


def mean_filling(df: pd.DataFrame) ->pd.DataFrame:
    df_clean = df.copy()
    print(f"total null values: {df_clean.isna().sum()}")

    df_clean = df_clean.fillna(df_clean.mean(numeric_only=True))
        

    print('Done mean fill!')
    return df_clean


def median_filling(df: pd.DataFrame) ->pd.DataFrame:
    df_clean = df.copy()
    print(f"total null values: {df_clean.isna().sum()}")

    df_clean = df_clean.fillna(df_clean.median(numeric_only=True))
        

    print('Done median fill!')
    return df_clean


def mode_filling(df: pd.DataFrame, datatype: int=0) ->pd.DataFrame:
    df_clean = df.copy()
    print(f"total null values: {df_clean.isna().sum()}")

    numeric_columns = df_clean.select_dtypes(include=['number']).columns

    non_numeric_columns = df_clean.select_dtypes(exclude=['number']).columns
    
    #numerical
    if datatype == 0:
        if not numeric_columns.empty:
            modes = df_clean[numeric_columns].mode().iloc[0]
            df_clean[numeric_columns] = df[numeric_columns].fillna(modes)
    
    #categorical
    else:
        if not non_numeric_columns.empty:
            modes = df_clean[non_numeric_columns].mode().iloc[0]
            df_clean[non_numeric_columns] = df_clean[non_numeric_columns].fillna(modes)

    print('Done mode fill!')
    return df_clean


def interpolate_fill(df: pd.DataFrame, axix: int) ->pd.DataFrame:
    df_clean = df.copy()
    print(f"total null values: {df_clean.isna().sum()}")
    
    numeric_columns = df_clean.select_dtypes(include=['number']).columns

    if axix ==1:
        df_clean[numeric_columns] = df[numeric_columns].interpolate(axis=1)
    
    else:
        df_clean[numeric_columns] = df[numeric_columns].interpolate(axis=0)

    print('Done interpolate fill!')
    return df_clean