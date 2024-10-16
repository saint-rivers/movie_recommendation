import joblib

model = joblib.load('save/recommender.pkl')
pred = model.predict(15235.0, 862.0)
print(pred)
