import numpy as np
import pickle
from flask import Flask, request, render_template
import joblib

# Load ML model
model = pickle.load(open('heart/randomforest_classifier_model.pkl', 'rb'))
scaler = joblib.load("heart/scaler.save")
# Create application
def predict(form):
    age = float(form['age'])
    sex = float(form['sex'])
    trestbps = float(form['trestbps'])
    chol = float(form['chol'])
    restecg = float(form['restecg'])
    thalach = float(form['thalach'])
    exang = float(form['exang'])
    cp = float(form['cp'])
    fbs = float(form['fbs'])
    oldpeak = float(form['oldpeak'])
    slope = float(form['slope'])
    ca = float(form['ca'])
    thal = float(form['thal'])

    array_features = np.array([age, sex, cp, trestbps, chol, fbs, restecg,
                               thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)

    # Convert features to array
    # Predict features
    prediction = model.predict(array_features)
    output = prediction
    # Check the output values and retrive the result with html tag based on the value
    return output
    # if output == 1:
    #     return 'The patient is not likely to have heart disease!', output=1)
    # else:
    #     return render_template('result.html',
    #                            result='The patient is likely to have heart disease!', output=0)
