# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# pip install tensorflow==2.16.1
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
import tensorflow.keras.backend as K


# %%
def recall_m(y_true, y_pred, batch_size=1024):
    def process_batch(start, end):
        batch_y_true = y_true[start:end]
        batch_y_pred = y_pred[start:end]
        true_positives = K.sum(K.round(K.clip(batch_y_true * batch_y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(batch_y_true, 0, 1)))
        return true_positives, possible_positives
    num_samples = y_true.shape[0]
    true_positives_total = 0
    possible_positives_total = 0
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch_true_positives, batch_possible_positives = process_batch(start, end)
        true_positives_total += batch_true_positives
        possible_positives_total += batch_possible_positives
    recall = true_positives_total / (possible_positives_total + K.epsilon())
    return recall


def precision_m(y_true, y_pred, batch_size=1024):
    def process_batch(start, end):
        batch_y_true = y_true[start:end]
        batch_y_pred = y_pred[start:end]
        true_positives = K.sum(K.round(K.clip(batch_y_true * batch_y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(batch_y_pred, 0, 1)))
        return true_positives, predicted_positives
    num_samples = y_true.shape[0]
    true_positives_total = 0
    predicted_positives_total = 0
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch_true_positives, batch_predicted_positives = process_batch(start, end)
        true_positives_total += batch_true_positives
        predicted_positives_total += batch_predicted_positives
    precision = float(true_positives_total) / float(predicted_positives_total + K.epsilon())
    return precision


def f1_m(y_true, y_pred, batch_size=1024):
    precision = precision_m(y_true, y_pred, batch_size)
    recall = recall_m(y_true, y_pred, batch_size)
    f1_score = 2 * ((precision * recall) / (precision + recall + K.epsilon()))
    return f1_score

# %%
model = tf.keras.models.load_model("test3.keras")

# %%
def predict(model, X_test):
    X_test = [X_test[:, 0], X_test[:, 1]]
    return model.predict(X_test)

# %%
df = pd.read_csv("./data/item_ratings.csv")#.iloc[:1000,:]
X = df[['userId','movieId']].to_numpy()
y = df['rating'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
num_users = df['userId'].max() + 1
num_items = df['movieId'].max() + 1

# %%
y_pred = predict(model, X_test)

# %%
recall = recall_m(y_test, y_pred)
precision = precision_m(y_test, y_pred)
f1 = f1_m(y_test, y_pred)

loss = model.evaluate()

with open('metrics.txt', 'w') as f:
    # Write the dictionary to the file in JSON format
    metrics = {
        # 'loss': loss,
        # 'accuracy': accuracy,
        'f1-score': f1,
        'precision': precision,
        'recall': recall
        }
    json.dump(metrics, f)
