import joblib
import kagglehub
import pandas as pd

model = joblib.load('save/recommender.pkl')
# pred = model.predict(15235.0, 862.0)
# model.get_neighbors(862.0, k =10)

test_movie_id = model.trainset.to_inner_iid(862.0)
n = model.get_neighbors(test_movie_id, k =10)
print(n)


df = pd.read_csv("./data/movies.csv", low_memory=False)

for mid in n:
    raw_id = model.trainset.to_raw_iid(mid)
    movie = df.loc[df['movieId']==raw_id]
    print(movie)
