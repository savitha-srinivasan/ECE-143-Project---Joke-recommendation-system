import data_cleaning
import normalization
import pandas as pd
import numpy as np 
import ubcf
import content_based_recommender
import sklearn 
from sklearn.feature_extraction.text import TfidfVectorizer

data_dir1 = "./Dataset Version 1/jester-data-1.xls"
data_dir2 = "./Dataset Version 1/jester-data-2.xls"
data_dir3 = "./Dataset Version 1/jester-data-3.xls"
df1 = pd.read_excel(data_dir1, header=None)
df2 = pd.read_excel(data_dir2, header=None)
df3 = pd.read_excel(data_dir3, header=None)
df = df1.append(df2).append(df3)
df = data_cleaning.clean_data(df)
normalized_df = normalization.normalize_data(df)

#test user
new_user = normalized_df.iloc[0].copy() 

#ubcf 
most_similar, most_similar_idx = ubcf.pearson(new_user,normalized_df)
top_user_matrix = normalized_df.iloc[most_similar_idx].copy()
top_5 = ubcf.top_similar_jokes(most_similar, top_user_matrix)
print("Top 5 recommended jokes using UBCF:", np.array(top_5)+1)
print("Actual rankings given by the user:\n{}".format(df.iloc[0, list(top_5)]))

#content-based recommender 
jokes_dir = "jester_content.xlsx"
df_joke = pd.read_excel(jokes_dir, header=None)
num_jokes, _ = df_joke.values.shape
joke_corpus = []
for index in range(num_jokes):
    joke = df_joke.values[index, 0]
    joke_corpus.append(joke)
tf_idf_vectorizer = TfidfVectorizer(use_idf=True)
tf_idf_matrix = tf_idf_vectorizer.fit_transform(joke_corpus)
recommended_jokes, predicted_rating = content_based_recommender.content_based(new_user, tf_idf_matrix.toarray(), tf_idf_matrix.toarray(), 5)
print("Top 5 recommended jokes using content based recommender:",np.array(recommended_jokes)+1)
print("Actual rankings given by the user:\n{}".format(df.iloc[0, recommended_jokes]))
