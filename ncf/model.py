import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='max',
    save_freq="epoch",
    save_best_only=True)

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
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mean_absolute_error')
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[model_checkpoint_callback])
        self.model = model
    def predict(self, X_test):
        X_test = [X_test[:, 0], X_test[:, 1]]
        return self.model.predict(X_test)