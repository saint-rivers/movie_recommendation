import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# preprocess movie data
# create a matrix of frequency-importance of words found in the overview
df = pd.read_csv("../data/movies.csv", low_memory=False)
tfidf = TfidfVectorizer(stop_words="english") #Frequency-Inverse Document Frequency
df['overview'] = df['overview'].fillna('')
df.to_csv("movies_processed.csv")
tfidf_matrix = tfidf.fit_transform(df['overview'])

# implement SVM
# linear_kernel is the simplest SVM kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
np.save("sim_matrix.npy", cosine_sim)
