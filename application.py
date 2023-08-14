from flask import Flask,request,render_template
import pandas as pd
import numpy as np 
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

#using flask api

application = Flask(__name__)

app = application

#route for the home page
@app.route('/')

def index():
    return render_template('index.html')

@app.route('/predictdatas', methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(YearMade=request.form.get('YearMade'),
                          fiSecondaryDesc=request.form.get('fiSecondaryDesc'),
                          fiProductClassDesc=request.form.get('fiProductClassDesc'),
                          saleDate=request.form.get('saledate'),
                          ModelID=request.form.get('ModelID'),
                          Track_Type=request.form.get('Track_Type'),
                          fiModelDesc=request.form.get('fiModelDesc'),
                          Enclosure=request.form.get('Enclosure'),
                          SalesID=request.form.get('SalesID'),
                          ProductSize=request.form.get('ProductSize'),
                          Tire_Size=request.form.get('Tire_Size'),
                          MachineID=request.form.get('MachineID'),
                          fiBaseModel=request.form.get('fiBaseModel'),
                          Coupler_System=request.form.get('Coupler_System'),
                          state=request.form.get('state'),
                          Ripper=request.form.get('Ripper'),
                          Transmission=request.form.get('Transmission'))
        pred_df = data.get_data_As_dataframe()
        print(f'Prediction data: \n {pred_df}')

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        print(f'Results: {results}')

        return render_template('home.html', results=results[0])
    
if __name__=='__main__':
    app.run(host='0.0.0.0',debug=True)