import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from database import find_movie_id

df = pd.read_csv("data/movies_processed_cleaned.csv", low_memory=False)
cosine_sim = np.load("nparray/overview_sim_matrix.npy")

# get indices of movie title
indices = pd.Series(df.index, index=df['id']).drop_duplicates()

def  get_recommendations(id, cosine_sim=cosine_sim):
    idx = indices[id]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1] , reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df[['title', 'id', 'runtime']].iloc[movie_indices]


# print(find_movie_id('lord'))
print("overview cosine")
print(get_recommendations(123))


print("feature cosine")
cos_sim_features = np.load("nparray/feature_sim_matrix.npy")
df = df.reset_index()
indices = pd.Series(df.index, index=df['id'])
print(get_recommendations(123, cosine_sim=cos_sim_features))
