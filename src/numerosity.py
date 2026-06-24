import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

def random_sampling(df:pd.DataFrame, fract:float=0.5) -> pd.DataFrame:
    df_clean = df.copy()
    return df_clean.sample(frac=fract)


def stratified_sampling(df:pd.DataFrame, column_name: str, fract:float=0.5) -> pd.DataFrame:
    df_clean = df.copy()
    return df_clean.groupby(column_name, group_keys=False).sample(frac=fract)


def cluster_reduction(df:pd.DataFrame, no_of_clusters:int)-> pd.DataFrame:
    df_clean = df.copy()
    kmeans = KMeans(n_clusters=no_of_clusters, random_state=42)
    numerical_columns = df_clean.select_dtypes(include=['number'])
    kmeans.fit(numerical_columns)

    return pd.DataFrame(kmeans.cluster_centers_, columns=numerical_columns.columns)


def get_regression_model(df: pd.DataFrame, x_col: list, y_col: str):
    X = df[x_col].values
    y = df[y_col].values
    model = LinearRegression().fit(X, y)
    return model.coef_, model.intercept_