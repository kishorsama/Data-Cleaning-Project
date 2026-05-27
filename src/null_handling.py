import pandas as pd

#drop na
def drop(df: pd.DataFrame, axix: int) ->pd.DataFrame:
    df_clean = df.copy()
    print(f"total null values: {df_clean.isna().sum()}")

    #column
    if axix == 1: 
        df_clean = df_clean.dropna(axis=1)
    #rows
    else:
        df_clean = df_clean.dropna(axis=0)

    print('Done drop!')
    return df_clean

#forward fill
def forward_fill(df: pd.DataFrame, axix: int) ->pd.DataFrame:
    df_clean = df.copy()
    print(f"total null values: {df_clean.isna().sum()}")

    if axix == 1: 
        df_clean = df_clean.ffill(axis=1)
        df_clean = df_clean.bfill(axis=1)
    #rows
    else:
        df_clean = df_clean.ffill(axis=0)
        df_clean = df_clean.bfill(axis=0)

    print('Done forward fill!')
    return df_clean

#backward fill
def backward_fill(df: pd.DataFrame, axix: int) ->pd.DataFrame:
    df_clean = df.copy()
    print(f"total null values: {df_clean.isna().sum()}")

    if axix == 1: 
        df_clean = df_clean.bfill(axis=1)
        df_clean = df_clean.ffill(axis=1)
        
    #rows
    else:
        df_clean = df_clean.bfill(axis=0)
        df_clean = df_clean.ffill(axis=0)
        

    print('Done backward fill!')
    return df_clean

#mean fill
def mean_filling(df: pd.DataFrame) ->pd.DataFrame:
    df_clean = df.copy()
    print(f"total null values: {df_clean.isna().sum()}")

    #only numeric ones
    df_clean = df_clean.fillna(df_clean.mean(numeric_only=True))
        

    print('Done mean fill!')
    return df_clean

#median fill
def median_filling(df: pd.DataFrame) ->pd.DataFrame:
    df_clean = df.copy()
    print(f"total null values: {df_clean.isna().sum()}")

    # only numeric
    df_clean = df_clean.fillna(df_clean.median(numeric_only=True))
        

    print('Done median fill!')
    return df_clean

# mode filling
def mode_filling(df: pd.DataFrame, datatype: int=0) ->pd.DataFrame:
    df_clean = df.copy()
    print(f"total null values: {df_clean.isna().sum()}")

    # numeric colums taken
    numeric_columns = df_clean.select_dtypes(include=['number']).columns
    # numeric colums taken
    non_numeric_columns = df_clean.select_dtypes(exclude=['number']).columns
    
    #numerical
    if datatype == 0:
        # make list of modes and take only non empty
        if not numeric_columns.empty:
            # then first value is used for rest filling
            modes = df_clean[numeric_columns].mode().iloc[0]
            df_clean[numeric_columns] = df[numeric_columns].fillna(modes)
    
    #categorical
    else:
        #same as above just categorical value
        if not non_numeric_columns.empty:
            modes = df_clean[non_numeric_columns].mode().iloc[0]
            df_clean[non_numeric_columns] = df_clean[non_numeric_columns].fillna(modes)

    print('Done mode fill!')
    return df_clean


#interpolate filling
def interpolate_fill(df: pd.DataFrame, axix: int) ->pd.DataFrame:
    df_clean = df.copy()
    print(f"total null values: {df_clean.isna().sum()}")

    #only numerical columns
    numeric_columns = df_clean.select_dtypes(include=['number']).columns

    if axix ==1:
        df_clean[numeric_columns] = df[numeric_columns].interpolate(axis=1)
    
    else:
        df_clean[numeric_columns] = df[numeric_columns].interpolate(axis=0)

    print('Done interpolate fill!')
    return df_clean