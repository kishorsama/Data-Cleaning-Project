import pandas as pd
from typing import List, Optional

# drop
def drop(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    df_clean = df.copy()
    if columns:
        valid_cols = [c for c in columns if c in df_clean.columns]
        if valid_cols:
            df_clean = df_clean.dropna(subset=valid_cols)
    else:
        df_clean = df_clean.dropna()
    return df_clean

# forward fill
def forward_fill(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    df_clean = df.copy()
    if columns:
        valid_cols = [c for c in columns if c in df_clean.columns]
        if valid_cols:
            df_clean[valid_cols] = df_clean[valid_cols].ffill()
    else:
        df_clean = df_clean.ffill()
    return df_clean

# backward fill
def backward_fill(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    df_clean = df.copy()
    if columns:
        valid_cols = [c for c in columns if c in df_clean.columns]
        if valid_cols:
            df_clean[valid_cols] = df_clean[valid_cols].bfill()
    else:
        df_clean = df_clean.bfill()
    return df_clean

# mean fill
def mean_filling(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    df_clean = df.copy()
    if columns:
        numeric_cols = df_clean[columns].select_dtypes(include='number').columns
        if not numeric_cols.empty:
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
    else:
        df_clean = df_clean.fillna(df_clean.mean(numeric_only=True))
    return df_clean

# median fill
def median_filling(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    df_clean = df.copy()
    if columns:
        numeric_cols = df_clean[columns].select_dtypes(include='number').columns
        if not numeric_cols.empty:
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
    else:
        df_clean = df_clean.fillna(df_clean.median(numeric_only=True))
    return df_clean

# mode filling
def mode_filling(df: pd.DataFrame, datatype: int = 0, columns: list = None) -> pd.DataFrame:
    df_clean = df.copy()
    if datatype == 0:
        target_cols = columns if columns is not None else df_clean.select_dtypes(include=['number']).columns
        target_cols = [c for c in target_cols if c in df_clean.columns]
        if target_cols:
            mode_df = df_clean[target_cols].mode()
            if not mode_df.empty:
                df_clean[target_cols] = df_clean[target_cols].fillna(mode_df.iloc[0])
    else:
        target_cols = columns if columns is not None else df_clean.select_dtypes(exclude=['number']).columns
        target_cols = [c for c in target_cols if c in df_clean.columns]
        if target_cols:
            mode_df = df_clean[target_cols].mode()
            if not mode_df.empty:
                df_clean[target_cols] = df_clean[target_cols].fillna(mode_df.iloc[0])
    return df_clean

# interpolate filling
def interpolate_fill(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    df_clean = df.copy()
    if columns:
        valid_cols = [c for c in columns if c in df_clean.columns]
        numeric_valid = df_clean[valid_cols].select_dtypes(include="number").columns
        if len(numeric_valid) > 0:
            df_clean[numeric_valid] = df_clean[numeric_valid].interpolate()
    else:
        numeric = df_clean.select_dtypes(include="number").columns
        df_clean[numeric] = df_clean[numeric].interpolate()
    return df_clean

# Smart Autonomous Dispatcher
def handle_nulls(df: pd.DataFrame, strategy: str = "auto", columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Unified, robust null-handling interface for Agentic and manual execution.
    Strategies: 'auto', 'mean', 'median', 'mode', 'drop', 'interpolate', 'ffill', 'bfill'
    """
    df_clean = df.copy()
    target_cols = columns if columns else [c for c in df_clean.columns if df_clean[c].isnull().sum() > 0]
    target_cols = [c for c in target_cols if c in df_clean.columns]

    if not target_cols:
        return df_clean

    strategy = (strategy or "auto").lower()

    if strategy == "drop":
        return drop(df_clean, target_cols)
    elif strategy == "mean":
        return mean_filling(df_clean, target_cols)
    elif strategy == "median":
        return median_filling(df_clean, target_cols)
    elif strategy == "mode":
        # Mode fill categorical + numerical
        num_cols = df_clean[target_cols].select_dtypes(include='number').columns.tolist()
        cat_cols = [c for c in target_cols if c not in num_cols]
        if num_cols:
            df_clean = mode_filling(df_clean, datatype=0, columns=num_cols)
        if cat_cols:
            df_clean = mode_filling(df_clean, datatype=1, columns=cat_cols)
        return df_clean
    elif strategy in ("ffill", "forward_fill"):
        return forward_fill(df_clean, target_cols)
    elif strategy in ("bfill", "backward_fill"):
        return backward_fill(df_clean, target_cols)
    elif strategy in ("interpolate", "interpolation"):
        return interpolate_fill(df_clean, target_cols)
    elif strategy == "auto":
        for col in list(target_cols):
            if col not in df_clean.columns:
                continue
            null_pct = df_clean[col].isnull().mean()
            # If >75% missing, drop the column
            if null_pct > 0.75:
                df_clean = df_clean.drop(columns=[col])
                continue

            if pd.api.types.is_numeric_dtype(df_clean[col]):
                # If skewed, use median, else mean
                skew = df_clean[col].skew()
                if abs(skew) > 1.0:
                    med = df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(med)
                else:
                    avg = df_clean[col].mean()
                    df_clean[col] = df_clean[col].fillna(avg)
            else:
                # Mode or "Unknown"
                modes = df_clean[col].mode()
                fill_val = modes.iloc[0] if not modes.empty else "Unknown"
                df_clean[col] = df_clean[col].fillna(fill_val)

    return df_clean
