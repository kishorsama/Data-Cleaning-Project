import pandas as pd
from typing import List, Optional

def handle_outliers(
    df: pd.DataFrame, 
    columns: Optional[List[str]] = None, 
    threshold: float = 1.5
) -> pd.DataFrame:
    """
    Handles numeric outliers using IQR clipping / winsorization.
    """
    df_clean = df.copy()
    
    if columns:
        num = [c for c in columns if c in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[c])]
    else:
        num = df_clean.select_dtypes(include=['number']).columns.tolist()

    for col in num:
        if df_clean[col].dropna().empty:
            continue
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR == 0:
            continue

        factor = threshold if threshold > 0 else 1.5
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR

        df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)

    return df_clean
