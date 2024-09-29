from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle

# Load the cleaned data
car = pd.read_csv('clean_data.csv')

# Model 
model=pickle.load(open("LinearRegressionModel.pkl",'rb'))



app = Flask(__name__)

@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = car['fuel_type'].unique()

    return render_template('index.html', companies=companies, car_models=car_models, years=years, fuel_types=fuel_types)

@app.route("/predict", methods=['POST'])
def predict():
    # Get form data
    selected_company = request.form.get('company')
    selected_model = request.form.get('model')
    selected_year = request.form.get('year')
    selected_fuel = request.form.get('fuel')
    kilometers_driven = request.form.get('kilo_driven')


    prediction=model.predict(pd.DataFrame([[selected_model,selected_company,selected_year,kilometers_driven,selected_fuel]],columns=['name','company','year','kms_driven','fuel_type']))
    print(prediction)
    # return f"Selected Company: {selected_company}, Model: {selected_model}, Year: {selected_year}, Fuel: {selected_fuel}, Kilometers: {kilometers_driven}"
    return jsonify({'price':str(np.round(prediction[0],2))})
if __name__ == "__main__":
    app.run(debug=True)
