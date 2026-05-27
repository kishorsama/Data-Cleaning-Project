import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def min_max_scaling(df: pd.DataFrame) -> pd.DataFrame:

    df_clean = df.copy()

    #only numeric columns
    numeric_columns = df_clean.select_dtypes(include=['number']).columns

    #if no numeric columns return as it is
    if numeric_columns.empty:
        print("No numeric values")
        return df_clean

    mmscaler = MinMaxScaler()
    df_clean[numeric_columns] = mmscaler.fit_transform(df_clean[numeric_columns])
    
    return df_clean