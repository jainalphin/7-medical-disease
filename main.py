import numpy as np
from flask import Flask, request, render_template
import Heart
import Cancer
import Kidney
import Liver
import Diabetes
import Malaria
import Pneumonia

# Create application
app = Flask(__name__)

# Bind home function to URL
@app.route("/")
@app.route("/home")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/breast_cancer",methods=['POST',"GET"])
def breast_cancer():
    if request.method =='GET':
        about = request.args.get('about', default='', type=str)
        if about:
            return render_template('breast_cancer/breast_cancer_about.html')

        return render_template("breast_cancer/breast_cancer.html")
    elif request.method=='POST':
        form = request.form
        output=Cancer.predict(form)
        if output == 0:
            return  render_template('result.html',result='The patient is likely to have Malignant!', output=1)
        else:
            return render_template('result.html',
                                   result='The patient is likely to have Benign!', output=0)

@app.route("/heart",methods=['POST',"GET"])
def heart():
    if request.method =='GET':
        about = request.args.get('about', default='', type=str)
        if about:
            return render_template('heart/heart_about.html')
        return render_template("heart/heart.html")
    elif request.method=='POST':
        form = request.form
        output=Heart.predict(form)
        if output == 1:
            return  render_template('result.html',result='The patient is not likely to have heart disease!', output=0)
        else:
            return render_template('result.html',
                                   result='The patient is likely to have heart disease!', output=1)

@app.route("/diabetes",methods=['GET', 'POST'])
def diabetes():
    if request.method =='GET':
        about = request.args.get('about', default='', type=str)
        if about:
            return render_template('diabetes/diabetes_about.html')
        return render_template("diabetes/diabetes.html")
    elif request.method=='POST':
        form = request.form
        output=Diabetes.predict(form)
        if output == 0:
            return  render_template('result.html',result='The patient is not likely to have Diabetes!', output=0)
        else:
            return render_template('result.html',
                                   result='The patient is likely to have Diabetes!', output=1)

@app.route("/liver",methods=['GET', 'POST'])
def liver():
    if request.method =='GET':
        about = request.args.get('about', default='', type=str)
        if about:
            return render_template('liver/liver_about.html')
        return render_template("liver/liver.html")
    elif request.method=='POST':
        form = request.form
        output=Liver.predict(form)
        if output == 0:
            return  render_template('result.html',result='The patient is not likely to have Liver disease)!', output=0)
        else:
            return render_template('result.html',
                                   result='The patient is likely to have Liver disease!', output=1)

@app.route("/kidney",methods=['GET', 'POST'])
def kidney():
    if request.method =='GET':
        about = request.args.get('about', default='', type=str)
        if about:
            return render_template('kidney/kidney_about.html')
        return render_template("kidney/kidney.html")
    elif request.method=='POST':
        form = request.form
        output=Kidney.predict(form)
        if output == 0:
            return render_template('result.html',result='The patient is not likely to have Kidney disease!', output=0)
        else:
            return render_template('result.html',
                                   result='The patient is likely to have Kidney disease!', output=1)

@app.route("/malaria",methods=['GET', 'POST'])
def malaria():
    if request.method =='GET':
        about = request.args.get('about', default='', type=str)
        if about:
            return render_template('malaria/malaria_about.html')
        return render_template("malaria/malaria.html")

    elif request.method=='POST':
        file= request.files['malaria']
        output= Malaria.predict(file)
        if output == 0 or output ==2:
            return render_template('result.html',
                                   result='The patient is not likely to have Malaria!', output=0)
        else:
            return render_template('result.html',
                                   result='The patient is likely to have Malaria!', output=1)


@app.route("/pneumonia",methods=['GET', 'POST'])
def pneumonia():
    if request.method =='GET':
        about = request.args.get('about', default='', type=str)
        if about:
            return render_template('pneumonia/pneumonia_about.html')
        return render_template("pneumonia/pneumonia.html")
    elif request.method=='POST':
        file= request.files['pneumonia']
        output= Pneumonia.predict(file)
        output = np.where(output>=0.5,1,0)
        if output == 0:
            return render_template('result.html',
                                   result='The patient is not likely to have Pneumonia!', output=0)
        else:
            return render_template('result.html',
                                   result='The patient is likely to have Pneumonia!', output=1)


if __name__ == "__main__":
    app.run(debug=True)
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

