import pandas as pd

# drop
def drop(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    df_clean = df.copy()
    print(f"Total null values before drop:\n{df_clean.isna().sum()}\n")

    if columns:
        df_clean = df_clean.dropna(subset=columns)
    else:
        df_clean = df_clean.dropna()

    print("Done Drop!")
    return df_clean

# forward fill
def forward_fill(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    df_clean = df.copy()
    print(f"Total null values before forward fill:\n{df_clean.isna().sum()}\n")

    if columns:
        # Removed .bfill() to strictly perform a forward fill
        df_clean[columns] = df_clean[columns].ffill() 
    else:
        df_clean = df_clean.ffill()

    print("Done Forward Fill!")
    return df_clean

# backward fill
def backward_fill(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    df_clean = df.copy()
    print(f"Total null values before backward fill:\n{df_clean.isna().sum()}\n")

    if columns:
         # Removed .ffill() to strictly perform a backward fill
        df_clean[columns] = df_clean[columns].bfill()
    else:
        df_clean = df_clean.bfill()

    print("Done Backward Fill!")
    return df_clean

# mean fill
def mean_filling(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    df_clean = df.copy()
    print(f"Total null values before mean fill:\n{df_clean.isna().sum()}\n")

    if columns:
        numeric_cols = df_clean[columns].select_dtypes(include='number').columns
        if not numeric_cols.empty:
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
    else:
        df_clean = df_clean.fillna(df_clean.mean(numeric_only=True))

    print("Done Mean Fill!")
    return df_clean

# median fill
def median_filling(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    df_clean = df.copy()
    print(f"Total null values before median fill:\n{df_clean.isna().sum()}\n")

    if columns:
        numeric_cols = df_clean[columns].select_dtypes(include='number').columns
        if not numeric_cols.empty:
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
    else:
        df_clean = df_clean.fillna(df_clean.median(numeric_only=True))

    print("Done Median Fill!")
    return df_clean

# mode filling
def mode_filling(df: pd.DataFrame, datatype: int = 0, columns: list = None) -> pd.DataFrame:
    df_clean = df.copy()
    print(f"Total null values before mode fill:\n{df_clean.isna().sum()}\n")

    # Numerical columns
    if datatype == 0:
        if columns is None:
            columns = df_clean.select_dtypes(include=['number']).columns
        
        mode_df = df_clean[columns].mode()
        if not mode_df.empty:
            modes = mode_df.iloc[0]
            df_clean[columns] = df_clean[columns].fillna(modes)

    # Categorical columns
    else:
        if columns is None:
            columns = df_clean.select_dtypes(exclude=['number']).columns

        mode_df = df_clean[columns].mode()
        
        if not mode_df.empty:
            modes = mode_df.iloc[0]
            df_clean[columns] = df_clean[columns].fillna(modes)

    print("Done Mode Fill!")
    return df_clean

# interpolate filling
def interpolate_fill(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    df_clean = df.copy()
    print(f"Total null values before interpolate fill:\n{df_clean.isna().sum()}\n")

    if columns:
        df_clean[columns] = df_clean[columns].interpolate()
    else:
        numeric = df_clean.select_dtypes(include="number").columns
        df_clean[numeric] = df_clean[numeric].interpolate()

    print("Done Interpolation!")
    return df_clean