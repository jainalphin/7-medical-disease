import numpy as np
import pickle
import joblib
# Load ML model
model = pickle.load(open('dibates/diabetes_rf_model.pkl', 'rb'))
scaler=joblib.load("dibates/scaler.save")
# Create application
def predict(form):
    size=len(form)
    to_predict_list = form.to_dict()
    to_predict_list = list(to_predict_list.values())
    to_predict_list = list(map(float, to_predict_list))
    to_predict = np.array(to_predict_list).reshape(1, -1)
    to_predict = scaler.transform(to_predict)
    prediction = model.predict(to_predict)
    output = prediction
    return output