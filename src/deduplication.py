import pandas as pd

def remove_duplicates(df:pd.DataFrame) -> pd.DataFrame:
    duplicate_count = df.duplicated().sum()
    duplicated_rows = df[df.duplicated()]
    
    df_clean = df.drop_duplicates()
    print("Duplicates Removed!")
    return df_clean
