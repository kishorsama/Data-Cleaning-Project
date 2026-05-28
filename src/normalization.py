import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def minmax(df: pd.DataFrame, cols: list) -> pd.DataFrame:
  df_clean = df.copy()
  scaler = MinMaxScaler()

  df_clean[cols] = scaler.fit_transform(df_clean[cols])

  print(f"Min-Max Normalization completed for columns: {cols}")
  return df_clean

