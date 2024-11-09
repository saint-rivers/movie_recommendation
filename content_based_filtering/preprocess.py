import kagglehub
import pandas as pd
import os
import json
import re
import ast
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import preprocessing


def parse_production_companies(x):
    if type(x) != str:
        return ""
    pattern  = r"[^a-zA-Z0-9\s]"
    try:
        g = ast.literal_eval(x)
        if len(g) <=0:
            return "[]"
        
        out = ""
        li = []
        for i, x in enumerate(g):
            name = x['name'].lower()
            name = re.sub(pattern, "", name)
            out += f"{name}"
            li.append(f"{name}")
            if i < len(g) - 1:
                out += ","
        payload = f"{out}"
        return out
    except:
        return ""
    
    
def genre_parser(data):
    g = data.replace("'", "\"")
    g = json.loads(g)
    li = []
    if len(g) > 0:
        out = ""
        for i, x in enumerate(g):
            out += f"{x['name'].lower()}"
            li.append(x['name'].lower())
            if i < len(g) - 1:
                out += ","
        return out
    else:
        return ""


def preprocess_production_companies(df: pd.DataFrame, col_name: str):
    df = df.drop(df.loc[df['production_companies'] == "[]"].index)
    df = df.drop(df.loc[df['production_companies'] == "False"].index)
    df = df.dropna(subset=['production_companies'])
    df.loc[df['production_companies'].isnull() == "[]"]
    return df


def load_data():
    path = kagglehub.dataset_download("rounakbanik/the-movies-dataset")
    return pd.read_csv(f"{path}/movies_metadata.csv", low_memory=False)


if __name__ == "__main__":
    df = load_data()
    df = df.drop(['production_countries', 'spoken_languages', 'belongs_to_collection', 'budget', 'poster_path', 'homepage', 'status', 'video', 'vote_count', 'vote_average'], axis=1)
    df = preprocess_production_companies(df, 'production_companies')
    df['production_companies'] = df['production_companies'].apply(parse_production_companies)
    df['genres'] = df['genres'].apply(genre_parser)
    df['overview'] = df['overview'].fillna('')
    df['title'] = df['title'].str.lower()
    df.to_csv("data/movies_processed_cleaned.csv")