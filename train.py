import joblib
from surprise import KNNWithMeans
from surprise.model_selection import cross_validate

# from recommender import model
from load_data import data

sim_options = {
    "name": "pearson_baseline",
    "user_based": False,  # Compute similarities between items
}
model = KNNWithMeans(sim_options=sim_options)
cv = cross_validate(model, data, measures = ['RMSE', 'MAE'], cv = 5, verbose =
True)



train_set = data.build_full_trainset()
model.fit(train_set)
joblib.dump(model, './save/item_based_recommender_knn.pkl')

# print(n)
# prediction = model.predict(1923.0, 8844.0)
# prediction.est