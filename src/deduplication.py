import pandas as pd

def remove_duplicates(df:pd.DataFrame) -> pd.DataFrame:
    df_clean = df.drop_duplicates()
    print("Duplicates Removed!")
    return df_clean
