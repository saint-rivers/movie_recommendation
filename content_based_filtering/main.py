import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

df = pd.read_csv("movies_processed.csv", low_memory=False)
cosine_sim = np.load("sim_matrix.npy")

# get indices of movie title
indices = pd.Series(df.index, index=df['original_title']).drop_duplicates()


def  get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1] , reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df['original_title'].iloc[movie_indices]


print(get_recommendations('Interstellar'))