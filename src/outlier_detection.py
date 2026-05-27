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

def zscore(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    df_clean = df.copy()
    cols = df_clean.select_dtypes(include=['number']).columns
    m = np.ones(len(df_clean), dtype=bool) # boolean mask
    
    for c in cols:
        c_data = df_clean[c].dropna() # drop NA for temp accurate calculation 
        z = np.abs(stats.zscore(c_data))
        
        c_m = z < threshold # keep rows based on condition
        valid_ind = c_data.index[c_m]
        
        # update main mask to keep rows valid
        temp_m = np.zeros(len(df_clean), dtype=bool) 
        temp_m[df_clean.index.isin(valid_ind)] = True
        m = m & temp_m
        
    print("Z-score Outliers Handled (Filtered).")
    return df_clean[m]

def isolation_forest(df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    df_clean = df.copy()
    col = df_clean.select_dtypes(include=['number']).columns
    
    # isolation required data without NaN for fitting
    df_temp = df_clean[col].dropna()
    
    forest = IsolationForest(contamination=contamination, random_state=42)
    pred = forest.fit_predict(df_temp)
    
    valid_ind = df_temp.iloc[pred == 1].index
    
    print("Isolation Forest Outliers Handled (Filtered).")
    return df_clean.loc[valid_ind]