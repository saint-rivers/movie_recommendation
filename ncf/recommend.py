# %%
import pandas as pd
import numpy as np
import tensorflow as tf

# %%
model = tf.keras.models.load_model("model/ncf_rec.keras")

# %%
df = pd.read_csv(f"data/item_ratings.csv")#.iloc[:1000,:]

# %%
user_id = 14524

seen_movies = df[df['userId'] == user_id]
movies_unneeded = df[df['movieId'].isin(seen_movies['movieId'])]
df_all = df.merge(movies_unneeded, on=['userId', 'movieId'], how="left", indicator=True)
unseen = df[df_all['_merge'] == 'left_only']
unseen_movie_ids = unseen['movieId'].unique()

# %%
unseen_movie_ids.shape[0]

# %%
p = np.full((unseen_movie_ids.shape[0],2), user_id)
p[:,1] = unseen_movie_ids
p

# %%
test_data = [p[:, 0], p[:, 1]]
out = model.predict(test_data)

# %%
np.unique(out, return_counts=True)

# %%
top_n = np.argsort(out.flatten())[-10:][::-1]
top_n

# %%
movies = pd.read_csv('data/movies_processed_cleaned.csv')
movies[movies['id'].isin(top_n)]


