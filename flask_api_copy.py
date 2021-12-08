from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
import flasgger
from flasgger import Swagger
app = Flask(__name__)
Swagger(app)
pickle_in = open('classifier.pkl','rb')
classifier = pickle.load(pickle_in)

## decorator
@app.route('/')
def welcome():
    return "welcome All"

@app.route('/predict', methods =["GET"])
def predict_note_authentication():
    """ Let's Authenticate the Bank Note
    ---
    parameters:
      - name : variance
        in : query
        type : number
        required : true
      - name : skewness
        in : query
        type : number
        required : true
      - name : curtosis
        in : query
        type : number
        required : true
      - name : entropy
        in : query
        type : number
        required : true
        
    
    responses:
        200:
            description: the output values
            
    """
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')        
            
    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])
    return "The predicted values is " + str(prediction)


@app.route('/predict_file', methods = ["POST"])
def predict_note_file():
    """ Let's Authenticate the Bank
    ---
    parameters:
      - name : file
        in : formData
        type : file
        required : true
        
    responses:
        200:
            description: the output values
            
    """        
    df_test = pd.read_csv(request.files.get('file'))
    pred = classifier.predict(df_test)
    return "The prediction values for the csv is " + str(list(pred))
    
if __name__ == '__main__':
    app.run()