import pandas as pd
from sklearn.preprocessing import LabelEncoder

def label_encoding(df: pd.DataFrame, cols: list):

    df_clean = df.copy()
    mappings = {}

    for c in cols:
        if c not in df_clean.columns:
            print(f"{c} not found")
            continue

        encoder = LabelEncoder()
        df_clean[c] = encoder.fit_transform(df_clean[c].astype(str))

        mappings[c] = dict(
            zip(
                encoder.classes_,
                encoder.transform(encoder.classes_)
            )
        )

    return df_clean, mappings