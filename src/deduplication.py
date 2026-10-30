import pandas as pd

def remove_duplicates(df: pd.DataFrame):
    """Legacy interface returning clean DataFrame, count, and duplicated rows."""
    duplicate_count = int(df.duplicated().sum())
    duplicated_rows = df[df.duplicated()].copy()
    df_clean = df.drop_duplicates().copy()
    return df_clean, duplicate_count, duplicated_rows

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Agent / Production interface returning directly the deduplicated DataFrame."""
    return df.drop_duplicates().reset_index(drop=True)
