import pandas as pd

def typecasting(df:pd.DataFrame) -> pd.DataFrame:
    df_clean=df.copy()
    print("\nAvailable Columns:")
    print(df_clean.columns.tolist())
    
    cols = input("\nEnter column names seperated by comma: ").split(',')
    type = input("\nEnter datatype (int, float, str): ")
    
    for c in cols:
        c=c.strip()
        if c in df_clean.columns:
            try:
                if type == 'int':
                    df_clean[c] = df_clean[c].astype('int')
                elif type == 'float':
                    df_clean[c] = df_clean[c].astype('float')
                elif type == 'str':
                    df_clean[c] = df_clean[c].astype('float')
                else:
                    print(f"Invalid datatype for column {c}")
            except Exception as e:
                print(f"Error converting {c}: {e}")
        else:
            print(f"Column '{c}' not found")
    print("\nType Casting Completed.")
    return df_clean
