from sklearn.preprocessing import MultiLabelBinarizer
import kagglehub
import pandas as pd
import os
import json


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
df = pd.read_csv(f"{path}/movies_metadata.csv", converters={'genres': genre_parser})
# print(os.listdir(path))


# reformat genres
mlb = MultiLabelBinarizer()
mlb_res = mlb.fit_transform([str(df.loc[i,'genres']).split(',') for i in range(
    len(df))])
df = df.join(pd.DataFrame(mlb_res,columns=list(mlb.classes_)))

# reformat original_language
# mlb_res = mlb.fit_transform([str(df.loc[i,'original_language']) for i in range(
#     len(df))])
# df = df.join(pd.DataFrame(mlb_res,columns=list(mlb.classes_)))


# remove unnecessary columns
df.drop(['belongs_to_collection', 'id'], axis=1, inplace=True)

print(df.columns)
