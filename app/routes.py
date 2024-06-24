from flask import render_template, request
from app import app
import pandas as pd
import pickle
from config import Config

# Load the dataset
df_1 = pd.read_csv("customer_data.csv")

# Define labels for input fields
input_labels = [
    'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 
    'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
    'PaymentMethod', 'tenure'
]

@app.route("/")
def load_page():
    return render_template('home.html', labels=input_labels, queries=[''] * 19)

@app.route("/", methods=['POST'])
def predict():
    # Get all input queries
    input_queries = {f'query{i}': request.form[f'query{i}'] for i in range(1, 20)}
    
    # Model selection
    model_choice = request.form['model_choice']
    model_path = Config.COMBINED_MODEL_PATH if model_choice == 'combined' else Config.BEST_MODEL_PATH
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    # Prepare the input data
    new_df = pd.DataFrame([list(input_queries.values())], columns=input_labels)
    
    df_2 = pd.concat([df_1, new_df], ignore_index=True)
    
    # Group the tenure in bins of 12 months
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    df_2['tenure_group'] = pd.cut(df_2.tenure.astype(int), range(1, 80, 12), right=False, labels=labels)
    df_2.drop(columns=['tenure'], axis=1, inplace=True)
    
    # Create dummy variables
    new_df_dummies = pd.get_dummies(df_2[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
       'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
       'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group']])
    
    # Make prediction
    prediction = model.predict(new_df_dummies.tail(1))
    probability = model.predict_proba(new_df_dummies.tail(1))[:,1]
    
    if prediction[0] == 1:
        output1 = "This customer is likely to churn!!"
    else:
        output1 = "This customer is likely to continue!!"
    
    output2 = f"Confidence: {probability[0]*100:.2f}%"
    
    return render_template('home.html', output1=output1, output2=output2, 
                           labels=input_labels, queries=list(input_queries.values()),
                           model_choice=model_choice)

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500