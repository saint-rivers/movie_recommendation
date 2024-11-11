import joblib
from surprise import KNNWithMeans
from surprise.model_selection import cross_validate
import pandas as pd
from surprise import Dataset
from surprise import Reader
import joblib
import kagglehub

df = pd.read_csv(f"data/item_ratings.csv")

ratings_dict = {
    "item": df['movieId'],
    "user": df['userId'],
    "rating": df['rating'],
}

df = pd.DataFrame(ratings_dict)
reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(df[["user", "item", "rating"]], reader)


sim_options = {
    "name": "pearson_baseline",
    "user_based": False,  # Compute similarities between items
}
model = KNNWithMeans(sim_options=sim_options)
cv = cross_validate(model, data, measures = ['RMSE', 'MAE'], cv = 5, verbose=True)


train_set = data.build_full_trainset()
model.fit(train_set)

import pickle
with open('models/item_based_collab.pkl', 'wb') as f:
    pickle.dump(model, f)

# print(n)
# prediction = model.predict(1923.0, 8844.0)
# prediction.est