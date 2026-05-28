import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def pca(df: pd.DataFrame, cols: list, n: int=2) -> pd.DataFrame:
    df_clean = df.copy()
    temp_data = df_clean[cols].dropna()
    
    p = PCA(n_components=n)
    res =p.fit_transform(temp_data)
    
    new_cols = [f"PCA_{i+1}" for i in range(n)] # generate new columns based on components required
    
    p_df =pd.DataFrame(res, columns=new_cols, index=temp_data.index) 
    
    df_clean = df_clean.drop(columns=cols).join(p_df)
    
    var=sum(p.explained_variance_ratio_)*100
    print(f"PCA Done! Kept about {round(var,1)}% of the variance.")
    return df_clean
    
def tsne(df: pd.DataFrame, cols: list, n: int=2, perp: float=30) -> pd.DataFrame:
    df_clean = df.copy()
    t = TSNE(n_components=n, perplexity=perp, random_state=42)
    
    valid_df = df_clean[cols].dropna()
    t_ft = t.fit_transform(valid_df)
    
    new_cols = [f"tSNE_{i+1}" for i in range(n)]
    
    temp = pd.DataFrame(t_ft, columns=new_cols, index=valid_df.index)
    
    df_clean = df_clean.drop(columns=cols).join(temp)
    print("tSNE finished.")
    
    return df_clean