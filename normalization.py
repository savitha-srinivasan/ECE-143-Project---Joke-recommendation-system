import pandas as pd 
import numpy as np 
def normalize_data(dataframe): 
    mean_data = dataframe.sum(axis=1, numeric_only=True)/dataframe.count(1)
    mean_sub = dataframe.subtract(mean_data, axis='rows')
    mean_pow = mean_sub.pow(2, axis='columns')
    var = np.sqrt(mean_pow.sum(axis=1, numeric_only=True)/(mean_pow.count(1)-1))
    return mean_sub.divide(var, axis='rows')