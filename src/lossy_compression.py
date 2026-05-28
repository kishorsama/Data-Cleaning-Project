import pandas as pd

def lossy(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    
    mem_start = df_clean.memory_usage(deep=True).sum() / 1024**2
    print(f"Starting memory: {mem_start:.2f} MB")
    
    for c in df_clean.columns:
        dtype = str(df_clean[c].dtype)
        
        if 'int' in dtype:
            df_clean[c] = pd.to_numeric(df_clean[c], downcast='integer')
            
        elif 'float' in dtype:
            df_clean[c] = pd.to_numeric(df_clean[c], downcast='float')
        
        elif dtype == 'object':
            u_count = len(df_clean[c].unique())
            t_count = len(df_clean[c])        
            
            if u_count / t_count <0.5:
                df_clean[c] = df_clean[c].astype('category')
    
    mem_end = df_clean.memory_usage(deep=True).sum() / 1024**2
    saved = 100 * (mem_start - mem_end) / mem_start
    
    print(f"Ending memory: {mem_end:.2f} MB")
    print(f"Saved {saved:.1f}%!")
    
    return df_clean