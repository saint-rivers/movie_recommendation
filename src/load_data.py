import pandas as pd
from surprise import Dataset
from surprise import Reader
import joblib

df = pd.read_csv("data/ratings_matrix.csv")

# This is the same data that was plotted for similarity earlier
# with one new user "E" who has rated only movie 1
ratings_dict = {
    "movie": df['movieId'],
    "user": df['userId'],
    "rating": df['rating'],
}

df = pd.DataFrame(ratings_dict)
reader = Reader(rating_scale=(0, 5))

# Loads Pandas dataframe
data = Dataset.load_from_df(df[["user", "movie", "rating"]], reader)
# Loads the builtin Movielens-100k data
movielens = Dataset.load_builtin('ml-100k')


