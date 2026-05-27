import pandas as pd
import numpy as np
import scipy as stats
from sklearn.ensemble import IsolationForest

def clipped(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    num = df_clean.select_dtypes(include=['number']).columns

    for col in num:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)

    print("IQR Outliers Handled (Clipped).")
    return df_clean

