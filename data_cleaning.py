import numpy as np 
import pandas as pd 

def clean_data(df): 
    df.index = range(df.shape[0])
    df.index.name = "User ID"
    col_names = ["joke"+str(i) for i in range(1, df.shape[1])]
    df.columns = ["NumRated"] + col_names
    df[df==99.0] = np.nan
    df = df.loc[:, df.columns != "NumRated"]
    return df
