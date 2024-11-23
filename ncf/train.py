import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import kagglehub
from model import NeuralCF



embedding_dim = 10
hidden_layers = [64, 32]
activation = 'relu'
learning_rate = 0.001

# Download latest version
# path = kagglehub.dataset_download("rishitjavia/netflix-movie-rating-dataset")
# we choose a subset of the netflix rating dataset found here : https://www.kaggle.com/datasets/rishitjavia/netflix-movie-rating-dataset?select=Netflix_Dataset_Rating.csv
# df = pd.read_csv(f"{path}/Netflix_Dataset_Rating.csv").iloc[:1000,:]
df = pd.read_csv(f"data/item_ratings.csv")#.iloc[:1000,:]
X = df[['userId','movieId']].to_numpy()
y = df['rating'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
num_users = df['userId'].max() + 1
num_items = df['movieId'].max() + 1

# Train and predict using NeuralCF
ncf = NeuralCF(num_users, num_items, embedding_dim, hidden_layers, activation, learning_rate)
ncf.train(X_train, y_train, epochs=10, batch_size=128)
y_pred = ncf.predict(X_test)

# Evaluate using Mean Squared Error
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
# accuracy_score = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)


user_id = 14524
# unseen_movies = [movie_id for movie_id in range(num_items) if movie_id not in df[df['userId'] == user_id]['movieId']]
# predictions = model.predict([np.full_like(unseen_movies, user_id), unseen_movies])
# top_n = np.argsort(predictions.flatten())[-10:][::-1]  # Top 10 movies with highest predicted ratings
# recommended_movies = movie_encoder.inverse_transform(top_n)
# print(recommended_movies)


seen_movies = df[df['userId'] == user_id]
movies_unneeded = df[df['movieId'].isin(seen_movies['movieId'])]
df_all = df.merge(movies_unneeded, on=['userId', 'movieId'], how="left", indicator=True)
unseen = df[df_all['_merge'] == 'left_only']
unseen_movie_ids = unseen['movieId'].unique()
# predictions = ncf.predict([np.full_like(unseen, user_id), unseen])
