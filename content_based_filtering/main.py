import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from content_based_filtering.database import find_movie_id
    

df = pd.read_csv("data/movies_processed_stopwords.csv", low_memory=False)
cosine_sim = np.load("nparray/overview_sim_matrix.npy")

# get indices of movie title
indices = pd.Series(df.index, index=df['movieId']).drop_duplicates()


def  get_recommendations(id, cosine_sim=cosine_sim):
    idx = indices[id]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1] , reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df[['title', 'movieId', 'runtime']].iloc[movie_indices]

    
print(find_movie_id('lord'))
print(get_recommendations(123))