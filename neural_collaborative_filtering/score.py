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


# %%
model = tf.keras.models.load_model("ncf.keras")