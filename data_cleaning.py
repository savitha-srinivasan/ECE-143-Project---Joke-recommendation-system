import numpy as np 
import pandas as pd 

def clean_data(df): 
    '''
    Function that cleans raw data; replaces 99.0 with NaN, removes unused columns and names columns based on joke number 
    :param df: dataframe object of user ratings for 100 jokes 
    '''
    assert isinstance(df, pd.core.frame.DataFrame)
    df.index = range(df.shape[0])
    df.index.name = "User ID"
    col_names = ["joke"+str(i) for i in range(1, df.shape[1])]
    df.columns = ["NumRated"] + col_names
    df[df==99.0] = np.nan
    df = df.loc[:, df.columns != "NumRated"]
    return df
