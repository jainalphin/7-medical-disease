import numpy as np
import pickle

# Load ML model
model = pickle.load(open('Breast_Cancer/breast_cancer_rf_model2.pkl', 'rb'))

# Create application
def predict(form):
    size=len(form)
    to_predict_list = form.to_dict()
    to_predict_list = list(to_predict_list.values())
    to_predict_list = list(map(float, to_predict_list))

    to_predict = np.array(to_predict_list).reshape(1, -1)
    prediction = model.predict(to_predict)
    output = prediction
    return output