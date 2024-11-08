import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from pathlib import Path
root = Path(__file__).parent.parent

# preprocess movie data
# create a matrix of frequency-importance of words found in the overview
df = pd.read_csv(f"{root}/data/movies.csv", low_memory=False)
df['overview'] = df['overview'].fillna('')
df['original_title'] = df['original_title'].str.lower()
df = df.drop('Unnamed: 0', axis=1)

df.to_csv("./data/movies_processed_stopwords.csv")

# implement SVM
# linear_kernel is the simplest SVM kernel
def generate_overview_matrix():
    tfidf = TfidfVectorizer(stop_words="english") #Frequency-Inverse Document Frequency
    tfidf_matrix = tfidf.fit_transform(df['overview'].values.astype('U'))
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    np.save("./nparray/overview_sim_matrix.npy", cosine_sim)

def get_producers(x):
    for i in x:
        return i['production_companies']['name']
    return np.nan


def generate_feature_sim_matrix():
    pass


if __name__ == "__main__":
    generate_overview_matrix()
