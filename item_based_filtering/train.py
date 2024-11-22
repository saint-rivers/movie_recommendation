import joblib
from surprise import KNNWithMeans
from surprise.model_selection import cross_validate
import pandas as pd
from surprise import Dataset
from surprise import Reader
import joblib
import kagglehub
from sklearn.metrics import f1_score, recall_score, accuracy_score
from surprise.model_selection import train_test_split

df = pd.read_csv(f"data/item_ratings.csv")

ratings_dict = {
    "item": df['movieId'],
    "user": df['userId'],
    "rating": df['rating'],
}

df = pd.DataFrame(ratings_dict)
reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(df[["user", "item", "rating"]], reader)
trainset, testset = train_test_split(data, test_size=0.25)

sim_options = {
    "name": "pearson_baseline",
    "user_based": False,  # Compute similarities between items
}
model = KNNWithMeans(sim_options=sim_options)
cv = cross_validate(model, data, measures = ['RMSE', 'MAE'], cv = 5, verbose=True)


trainset = trainset.build_full_trainset()
model.fit(trainset)

predictions = model.test(testset)

def print_f1():
    threshold = 3
    y_true = [int(true_r >= threshold) for (_, _, true_r, _, _) in predictions]
    y_pred = [int(est >= threshold) for (_, _, _, est, _) in predictions]

    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Print results
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

print_f1()

import pickle
with open('models/item_based_collab.pkl', 'wb') as f:
    pickle.dump(model, f)

# print(n)
# prediction = model.predict(1923.0, 8844.0)
# prediction.est