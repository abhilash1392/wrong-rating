import re
import numpy as np
from flask import Flask, request, jsonify, render_template,Response,send_file
import joblib
from test import predict
import pandas as pd
from test_form import predict_file

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_result',methods=['POST'])
def predict_result():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    review,rating = int_features[0],int(int_features[1])
    if rating>3:
        return render_template('index.html',review=review,rating=str(rating),prediction_text="This rating is already good. Don't you Worry")
    else:
        output = predict(review,rating)
        if output==True:
            return render_template('index.html',review=review,rating=str(rating),prediction_text="This rating is wrong. Please contact the customer")
        if output==False:
            return render_template('index.html',review=review,rating=str(rating),prediction_text="This rating is correct. Please contact the support team")
@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API call through reqyest
    '''
    print("Posted file: {}".format(request.files['file']))
    file = request.files['file']
    data = pd.read_csv(file)
    data.to_csv('data.csv')
    predict_file()
    return getCsv() 

# @app.route("/getCsv")
def getCsv():
    return send_file('savedFile.csv',mimetype="text/csv",attachment_filename='savedFile.csv',as_attachment=True)

@app.route('/upload_file')
def upload_file():
    return render_template("upload.html")

if __name__=="__main__":
    app.run(debug=True)

