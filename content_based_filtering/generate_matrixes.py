import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from pathlib import Path
from database import find_movie_id
root = Path(__file__).parent.parent

# preprocess movie data
# create a matrix of frequency-importance of words found in the overview


# implement SVM
# linear_kernel is the simplest SVM kernel
def generate_overview_matrix():
    tfidf = TfidfVectorizer(stop_words="english") #Frequency-Inverse Document Frequency
    tfidf_matrix = tfidf.fit_transform(df['overview'].values.astype('U'))
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    np.save("./nparray/overview_sim_matrix.npy", cosine_sim)


def check_empty_list(x):
    if x == "":
        return []
    elif x == None:
        return []
    else:
        return x


def create_soup(x: pd.Series):
    def join_feat(x):
        if x == None:
            parsed = ""
        else:
            out = str(x).split(",")
            parsed = ' '.join(out)
        return parsed
    
    out = join_feat(x['genres'])
    out = out + ' ' + join_feat(x['production_companies'])
    out = out + ' ' + join_feat(x['title'])
    return out


def generate_feature_matrix():
    df = pd.read_csv("data/movies_processed_cleaned.csv")
    df = df[['production_companies', 'genres', 'title', 'id']]
    df['soup'] = df.apply(create_soup, axis=1)
    
    count = CountVectorizer(stop_words="english")
    count_matrix = count.fit_transform(df['soup'])
    cos_sim = cosine_similarity(count_matrix, count_matrix)
    np.save("./nparray/feature_sim_matrix.npy", cos_sim)
    

    
if __name__ == "__main__":
    # generate_overview_matrix()
    generate_feature_matrix()