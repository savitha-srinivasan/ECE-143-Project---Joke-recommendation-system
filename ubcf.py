import numpy as np 
import pandas as pd 

#similarity computation
def pearson(new_user, dataframe): 
    '''
    Function that computes Pearson correlation between a new user and the dataframe of all user ratings, then finds the most similar users based on this score 
    :param new_user: Pandas series object of ratings for the new user 
    :param dataframe: Pandas Dataframe object containing ratings of all users for all jokes 
    '''
    assert isinstance(new_user, pd.core.series.Series)
    assert isinstance(dataframe, pd.core.frame.DataFrame)
    d1 = np.nansum(np.square(new_user)) 
    d2 = np.nansum(np.square(dataframe),axis=1)
    similar_users = np.nan_to_num(np.nansum(dataframe*new_user, axis=1)/np.sqrt(d1*d2))
    most_similar = np.sort(similar_users)[-20:]
    most_similar_idx = np.argsort(similar_users)[-20:]
    return (most_similar, most_similar_idx)

def top_similar_jokes(most_similar, top_user_matrix): 
    '''
    Function that computes the top 5 recommended jokes using the computed most similar users and their ratings 
    :param most_similar: numpy array of computed scores of most similar users based on Pearson correlation 
    :param top_user_matrix: dataframe object containing corresponding ratings for the most similar users 
    '''
    assert isinstance(most_similar, np.ndarray)
    assert isinstance(top_user_matrix, pd.core.frame.DataFrame)
    scores_matrix = top_user_matrix.copy() 
    top_jokes = np.nansum(top_user_matrix.T * most_similar, axis=1)/np.sum(most_similar)
    top_5 = np.argsort(top_jokes)[-5:]
    return top_5