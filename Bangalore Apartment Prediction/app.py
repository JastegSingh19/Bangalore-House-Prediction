from flask import Flask,render_template,redirect,request
import pandas as pd
import pickle
import numpy as np
# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)
data=pd.read_csv("Cleaned data.csv")
pipe=pickle.load(open("RidgeModel.pkl",'rb'))

# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
@app.route('/')
# ‘/’ URL is bound with hello_world() function.
def hello_world():
    locations=sorted(data['location'].unique())
    return render_template('index.html',locations=locations)
@app.route('/predict',methods=['POST'])
def predict():
    location=request.form.get('location')
    bhk=request.form.get('bhk')
    bath =float(request.form.get('bath'))
    sqft =float(request.form.get('sq_feet'))
    print(location,bhk,bath,sqft)
    input=pd.DataFrame([[location,sqft,bath,bhk]],columns=['location','total_sqft','bath','bhk'])
    prediction=pipe.predict(input)[0]*100000
    if(prediction<0):
        return "is wrong due to incorrect data"
    return str(np.round(prediction,2))


# main driver function
if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.
    app.run(debug=True,port=5000)
