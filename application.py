import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# import scaler pickle and linear, lasso and elatic regressor
# generic structure is as follows
# instance_name = pickle.load(open('path with file name','mode in which file to be opened'))

standard_scaler = pickle.load(open('models/scaler.pkl','rb'))
linear_model = pickle.load(open('models/linear.pkl','rb'))
lasso_model = pickle.load(open('models/lasso.pkl','rb'))
enet_model = pickle.load(open('models/elastic.pkl','rb'))


@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata', methods = ['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':
        Temperature=float(request.form.get('Temperature'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled=standard_scaler.transform([[Temperature,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result_linear=linear_model.predict(new_data_scaled)
        result_lasso=lasso_model.predict(new_data_scaled)
        result_enet=enet_model.predict(new_data_scaled)

        return render_template('home.html',result_linear=result_linear[0], result_lasso=result_lasso[0], result_enet=result_enet[0])

        
    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")
