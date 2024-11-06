import pickle

from flask import Flask, render_template
import json

model = None

app = Flask(__name__)
with open('save/item_based_recommender_knn.pkl', 'rb') as f:
    model = pickle.load(f)
    

@app.route('/submit', methods=['GET', 'POST'])  
def make_prediction():  
    if model == None:
        return json.dumps({'message':'model not loaded'})
    # features = [int(x) for x in request.form.values()]  
    # final_features = [np.array(features)]         
    # prediction = model.predict(final_features)    
    # prediction = prediction[0]      
    # return render_template('prediction.html', prediction = prediction) 
    return json.dumps({'message':'not implemented exception'})


@app.route("/")                        
def index():
     return render_template('index.html')