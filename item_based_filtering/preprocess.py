# %%
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import resample
import kagglehub


# %%
path = kagglehub.dataset_download("rounakbanik/the-movies-dataset")
ratings = pd.read_csv(f"{path}/ratings.csv").dropna()

# %%
ratings = ratings.drop(labels=['timestamp'], axis=1)

# %%
movies = pd.read_csv("../data/movies_processed_cleaned.csv")

# %%
df = pd.merge(movies, ratings, left_on="id", right_on="movieId", how="right")
df = df[['userId', 'id', 'rating']]
df['movieId'] = df['id']
df = df.drop(labels=['id'], axis=1)

# %%
df.to_csv("../data/item_ratings.csv")


