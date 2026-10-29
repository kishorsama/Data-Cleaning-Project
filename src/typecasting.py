import pandas as pd


def typecasting(df: pd.DataFrame, cols: list, dtype: str) -> pd.DataFrame:
    df_clean = df.copy()

    for col in cols:
        if col not in df_clean.columns:
            print(f"Column '{col}' not found")
            continue

        try:
            if dtype == 'datetime64[ns]':
                df_clean[col] = pd.to_datetime(df_clean[col])
            elif dtype == 'str':
                df_clean[col] = df_clean[col].astype('string')
            else:
                df_clean[col] = df_clean[col].astype(dtype)
        except Exception as e:
            print(f"Error converting {col}: {e}")

    print("\nType Casting Completed.")
    return df_clean
