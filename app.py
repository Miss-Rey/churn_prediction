# coding: utf-8

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics
from flask import Flask, request, render_template
import pickle
import os

app = Flask(__name__)

df_1 = pd.read_csv("customer_data.csv")

@app.route("/")
def loadPage():
    return render_template('home.html', query="")

@app.route("/", methods=['POST'])
def predict():
    input_fields = [
        'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender',
        'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'tenure'
    ]
    
    input_data = {field: request.form[f'query{i+1}'] for i, field in enumerate(input_fields)}
    
    selected_model = request.form['model_choice']
    model_path = os.path.join('models', f"{selected_model}.pkl")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    new_df = pd.DataFrame([input_data])
    
    df_2 = pd.concat([df_1, new_df], ignore_index=True) 
    
    # Group the tenure in bins of 12 months
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    df_2['tenure_group'] = pd.cut(df_2.tenure.astype(int), range(1, 80, 12), right=False, labels=labels)
    df_2.drop(columns=['tenure'], axis=1, inplace=True)   
    
    categorical_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                           'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                           'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                           'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group']
    
    new_df_dummies = pd.get_dummies(df_2[categorical_columns])
    
    single = model.predict(new_df_dummies.tail(1))
    probability = model.predict_proba(new_df_dummies.tail(1))[:,1]
    
    if single[0] == 1:
        o1 = "This customer is likely to churn!"
        o2 = f"Confidence: {probability[0]*100:.2f}%"
    else:
        o1 = "This customer is likely to continue!"
        o2 = f"Confidence: {(1-probability[0])*100:.2f}%"
        
    return render_template('home.html', output1=o1, output2=o2, 
                           **{f'query{i+1}': input_data[field] for i, field in enumerate(input_fields)},
                           model_choice=selected_model)

if __name__ == "__main__":
    app.run(debug=True)