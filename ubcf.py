import numpy as np 
import pandas as pd 

#similarity computation
def pearson(new_user, dataframe): 
    d1 = np.nansum(np.square(new_user)) 
    d2 = np.nansum(np.square(dataframe),axis=1)
    similar_users = np.nan_to_num(np.nansum(dataframe*new_user, axis=1)/np.sqrt(d1*d2))
    most_similar = np.sort(similar_users)[-20:]
    most_similar_idx = np.argsort(similar_users)[-20:]
    return (most_similar, most_similar_idx)

def top_similar_jokes(most_similar, top_user_matrix): 
    scores_matrix = top_user_matrix.copy() 
    top_jokes = np.nansum(top_user_matrix.T * most_similar, axis=1)/np.sum(most_similar)
    top_5 = np.argsort(top_jokes)[-5:]
    return top_5