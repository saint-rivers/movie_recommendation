import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
# from database import find_movie_id


# df = pd.read_csv("content_based_filtering/data/movies_processed_cleaned.csv", low_memory=False)
overview_sim_mat = np.load("content_based_filtering/nparray/overview_sim_matrix.npy")
feature_sim_mat = np.load("content_based_filtering/nparray/feature_sim_matrix.npy")
indices = pd.Series(df.index, index=df['id']).drop_duplicates()

def  get_recommendations(id: int, cosine_sim=feature_sim_mat):
    idx = indices[id]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1] , reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movie_indices
    # return df[['title', 'id', 'runtime']].iloc[movie_indices]
