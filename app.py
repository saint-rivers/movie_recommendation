import pickle

from flask import Flask, request, jsonify, render_template
import pandas as pd
import json

model = None

app = Flask(__name__)
with open('item_based_filtering/models/item_based_recommender_knn.pkl', 'rb') as f:
    model = pickle.load(f)
    
df = pd.read_csv("./data/movies.csv", low_memory=False)


@app.route('/search', methods=['GET'])
def search_movie():
    title = request.args['title']
    out = df.query(f"original_title.str.contains('{title}', case=False)")
    return out.to_dict('records')


@app.route('/recommend', methods=['POST'])  
def make_prediction():  
    if model == None:
        return jsonify({'message':'model not loaded'})
    
    movie_id = request.args['movie']

    test_movie_id = model.trainset.to_inner_iid(float(movie_id))
    n = model.get_neighbors(test_movie_id, k =10)

    ids = [model.trainset.to_raw_iid(mid) for mid in n]
    indexes = df.index.isin(ids)
    out = df[indexes]
    out = out[['adult', 'genres', 'original_language', 'original_title', 'overview', 'runtime', 'production_companies']]
    
    return jsonify({'payload': out.to_dict('records')})


@app.route('/pages/recommend', methods=['GET'])  
def recommend_page():  
    if model == None:
        return jsonify({'message':'model not loaded'})
    
    movie_id = request.args['movie']

    test_movie_id = model.trainset.to_inner_iid(float(movie_id))
    n = model.get_neighbors(test_movie_id, k =10)

    ids = [model.trainset.to_raw_iid(mid) for mid in n]
    indexes = df.index.isin(ids)
    out = df[indexes]
    out = out[['adult', 'genres', 'original_language', 'original_title', 'overview', 'runtime', 'production_companies']]
    
    # return jsonify({'payload': out.to_dict('records')})
    return render_template('recommend.html', output=out.to_dict('records'))


@app.route("/")                        
def index():
     return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)