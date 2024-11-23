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
import json
import tensorflow.keras.backend as K


# %%

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

class NeuralCF:
    def __init__(self, num_users, num_items, embedding_dim=10, hidden_layers=[64, 32], activation='relu', learning_rate=0.001):
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
    def _build_model(self):
        user_input = Input(shape=(1,))
        item_input = Input(shape=(1,))
        user_embedding = Embedding(self.num_users, self.embedding_dim)(user_input)
        user_embedding = Flatten()(user_embedding)
        item_embedding = Embedding(self.num_items, self.embedding_dim)(item_input)
        item_embedding = Flatten()(item_embedding)
        vector = Concatenate()([user_embedding, item_embedding])
        for units in self.hidden_layers:
            vector = Dense(units, activation=self.activation)(vector)
        output = Dense(1, activation='sigmoid')(vector)
        model = Model(inputs=[user_input, item_input], outputs=output)
        return model
    def train(self, X_train, y_train, epochs=10, batch_size=64, validation_split=0.1):
        X_train = [X_train[:, 0], X_train[:, 1]]
        y_train = np.array(y_train)
        model = self._build_model()
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=['mean_squared_error', 'root_mean_squared_error'], metrics=['acc',f1_m,precision_m, recall_m])
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        print(f'\n\n##### printing history #####\n')
        print(history)
        loss, accuracy, f1_score, precision, recall = model.evaluate(X_train, y_train, verbose=0)
        with open('metrics.txt', 'w') as f:
            # Write the dictionary to the file in JSON format
            metrics = {
                'loss': loss,
                'accuracy': accuracy,
                'f1-score': f1_score,
                'precision': precision,
                'recall': recall
            }
            json.dump(metrics, f)
        self.model = model
    def predict(self, X_test):
        X_test = [X_test[:, 0], X_test[:, 1]]
        return self.model.predict(X_test)
    def save(self):
        self.model.save("ncf.keras")


# %%
# we choose a subset of the netflix rating dataset found here : https://www.kaggle.com/datasets/rishitjavia/netflix-movie-rating-dataset?select=Netflix_Dataset_Rating.csv

df = pd.read_csv("data/item_ratings.csv")#.iloc[:1000,:]
X = df[['userId','movieId']].to_numpy()
y = df['rating'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
num_users = df['userId'].max() + 1
num_items = df['movieId'].max() + 1

# %%
import tensorflow
print(tensorflow.config.list_physical_devices('GPU'))

# %%
# Hyperparameters that could be tuned
embedding_dim = 10
hidden_layers = [64, 32]
activation = 'relu'
learning_rate = 0.001

# Train and predict using NeuralCF
ncf = NeuralCF(num_users, num_items, embedding_dim, hidden_layers, activation, learning_rate)
ncf.train(X_train, y_train)
ncf.save()

y_pred = ncf.predict(X_test)


from keras import backend as K
# Evaluate using Mean Squared Error
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true))) 

mse = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

print("Mean Absolute Error:", mse)


