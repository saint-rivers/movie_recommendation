from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import preprocessing
import kagglehub
import pandas as pd
import os
import json

def spoken_languages_parser(data):
    # g = data.replace("'", "\"")
    # g = json.loads(data)
    if len(data) > 0:
        out = ""
        for i, x in enumerate(data):
            out += f"{x['iso_639_1']}"
            if i < len(data) - 1:
                out += ","
        return out
    else:
        return None
    
def production_countries_parser(data):
    g = data.replace("'", "\"")
    g = json.loads(g)
    if len(g) > 0:
        out = ""
        for i, x in enumerate(g):
            out += f"{x['iso_3166_1']}"
            if i < len(g) - 1:
                out += ","
        return out
    else:
        return None
    
def genre_parser(data):
    g = data.replace("'", "\"")
    g = json.loads(g)
    if len(g) > 0:
        out = ""
        for i, x in enumerate(g):
            out += f"{x['name'].lower()}"
            if i < len(g) - 1:
                out += ","
        return out
    else:
        return None


# Download latest version
path = kagglehub.dataset_download("rounakbanik/the-movies-dataset")
converters = {
    'genres': genre_parser, 
    # 'production_countries': production_countries_parser, 
    # 'spoken_languages': spoken_languages_parser
}
df = pd.read_csv(f"{path}/movies_metadata.csv", converters=converters)
# print(os.listdir(path))


# reformat genres
# mlb = MultiLabelBinarizer()
# mlb_res = mlb.fit_transform([str(df.loc[i,'genres']).split(',') for i in range(
#     len(df))])
# df = df.join(pd.DataFrame(mlb_res,columns=list(mlb.classes_)))


rating_data = pd.read_csv(f"{path}/ratings.csv")

df['id'] = df['id'].astype(object)
s = pd.to_numeric(df['id'],errors='coerce')
df.loc[s.notna(), 'movieId'] = s.dropna().astype(float)
rating_data['movieId'] = rating_data['movieId'].astype(float)


df.drop(['belongs_to_collection', 'id', 'budget', 'poster_path', 'homepage', 'tagline', 'status', 'video', 'vote_count', 'vote_average'], axis=1, inplace=True)
table = pd.merge(df, rating_data, on="movieId", how="left")

# table.drop(['timestamp'], axis=1, inplace=True)
table.dropna(inplace=True)

# remove unnecessary columns
# table.to_csv('parsed.csv')

mat = table[['movieId', 'userId', 'rating']]
mat.sort_values(by="userId")

# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(mat.values)
# mat = pd.DataFrame(x_scaled, columns=mat.columns, index=mat.index)

mat.to_csv('./data/ratings_matrix.csv')
df.to_csv('./data/movies.csv')


print(mat)
