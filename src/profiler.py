import pandas as pd
from typing import Dict, Any

def get_dataset_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Inspects a Pandas DataFrame and extracts key statistical metadata 
    without sending raw data rows to the LLM. 
    This helps the AI agent decide which cleaning operations to run.
    """
    profile = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": {}
    }
    
    for col in df.columns:
        series = df[col]
        is_numeric = pd.api.types.is_numeric_dtype(series)
        null_count = int(series.isnull().sum())
        
        # Base metadata for every column type
        col_meta = {
            "dtype": str(series.dtype),
            "null_count": null_count,
            "null_percentage": round((null_count / len(df)) * 100, 2) if len(df) > 0 else 0.0,
            "unique_values": int(series.nunique()),
        }
        
        # Add distribution metrics only for numeric columns (crucial for choosing outlier strategies)
        if is_numeric and series.dropna().shape[0] > 0:
            col_meta["skewness"] = round(float(series.skew()), 2)
            col_meta["mean"] = round(float(series.mean()), 2)
            col_meta["median"] = round(float(series.median()), 2)
        else:
            col_meta["skewness"] = None
            
        profile["columns"][col] = col_meta
        
    return profile