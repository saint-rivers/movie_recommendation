import joblib
import kagglehub
import pandas as pd

model = joblib.load('models/item_based_collab.pkl')


def  get_recommendations(id: int):
    inner_id = model.trainset.to_inner_iid(id)
    neighbor_ids = model.get_neighbors(inner_id, k =10)
    ids = [model.trainset.to_raw_iid(x) for x in neighbor_ids]
    return ids