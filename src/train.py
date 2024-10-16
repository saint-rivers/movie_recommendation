import joblib
from recommender import algo
from load_data import data

train_set = data.build_full_trainset()
algo.fit(train_set)
joblib.dump(algo, 'save/recommender.pkl')

prediction = algo.predict(1923.0, 8844.0)
prediction.est