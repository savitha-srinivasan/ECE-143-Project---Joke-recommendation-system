import numpy as np 
import pandas as pd

#content-based system
def content_based(normalized_user, tf_idf_dict, unseen_jokes, n=5):
    user_profile = np.nanmean(tf_idf_dict * normalized_user[:, np.newaxis], axis=0)
    norm_unseen = np.sqrt(np.nansum(np.power(unseen_jokes, 2), axis=1))
    norm_user = np.sqrt(np.nansum(np.power(user_profile, 2)))
    #cosine similarity between user profile and unseen joke matrix
    dot_product = np.nansum(unseen_jokes * user_profile.T, axis=1)
    pred_rating = dot_product / (norm_unseen * norm_user)
    recommended_jokes = np.argsort(pred_rating)[-n:]
    return recommended_jokes, pred_rating