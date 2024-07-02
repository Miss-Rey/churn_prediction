# Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, render_template
import pickle
import os

# Initialize the Flask application
app = Flask(__name__)

# Load the initial dataset
df_1 = pd.read_csv("customer_data.csv")

# Define the home route
@app.route("/")
def loadPage():
    return render_template('home.html', query="")

# Define the route to handle form submission and prediction
@app.route("/", methods=['POST'])
def predict():
    # Define the input fields expected from the form
    input_fields = [
        'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender',
        'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'tenure'
    ]
    
    # Extract input data from the form
    input_data = {field: request.form[f'query{i+1}'] for i, field in enumerate(input_fields)}
    
    # Load the selected model
    selected_model = request.form['model_choice']
    model_path = os.path.join('models', f"{selected_model}.pkl")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    # Create a new DataFrame with the input data
    new_df = pd.DataFrame([input_data])
    
    # Concatenate the new data with the existing dataset
    df_2 = pd.concat([df_1, new_df], ignore_index=True) 
    
    # Group the tenure in bins of 12 months
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    df_2['tenure_group'] = pd.cut(df_2.tenure.astype(int), range(1, 80, 12), right=False, labels=labels)
    df_2.drop(columns=['tenure'], axis=1, inplace=True)   
    
    # Define the categorical columns for one-hot encoding
    categorical_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                           'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                           'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                           'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group']
    
    # Perform one-hot encoding
    new_df_dummies = pd.get_dummies(df_2[categorical_columns])
    
    # Make a prediction
    single = model.predict(new_df_dummies.tail(1))
    probability = model.predict_proba(new_df_dummies.tail(1))[:,1]
    
    # Determine the output message based on the prediction
    if single[0] == 1:
        o1 = "This customer is likely to churn!"
        o2 = f"Confidence: {probability[0]*100:.2f}%"
    else:
        o1 = "This customer is likely to continue!"
        o2 = f"Confidence: {(1-probability[0])*100:.2f}%"
        
    # Render the output on the HTML template
    return render_template('home.html', output1=o1, output2=o2, 
                           **{f'query{i+1}': input_data[field] for i, field in enumerate(input_fields)},
                           model_choice=selected_model)

# Run the Flask application
if __name__ == "__main__":
    app.run(debug=True)
