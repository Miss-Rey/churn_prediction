from flask import Flask, request, render_template, jsonify
import joblib
from apscheduler.schedulers.background import BackgroundScheduler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from imblearn.combine import SMOTEENN
from collections import Counter
import os
import time
import threading

app = Flask(__name__)

# Paths to the model files
best_model_path = 'best_individual_model.pkl'
combined_model_path = 'combined_model.pkl'

# Initialize global variables
best_model = None
combined_model = None
current_model = None
is_training = False

# Define the preprocessing steps for numerical and categorical features globally
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine preprocessing steps into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)],
    remainder='passthrough')

def load_models():
    global best_model, combined_model, current_model

    if os.path.exists(best_model_path):
        best_model = joblib.load(best_model_path)
        print(f"Best model loaded successfully. Type: {type(best_model)}")
        if isinstance(best_model, Pipeline) and hasattr(best_model[-1], 'predict'):
            print("Best model is fitted and ready for predictions")
        else:
            print("Warning: Best model is not properly fitted")
            best_model = None

    if os.path.exists(combined_model_path):
        combined_model = joblib.load(combined_model_path)
        print(f"Combined model loaded successfully. Type: {type(combined_model)}")
        if isinstance(combined_model, Pipeline) and hasattr(combined_model[-1], 'predict'):
            print("Combined model is fitted and ready for predictions")
        else:
            print("Warning: Combined model is not properly fitted")
            combined_model = None

    current_model = best_model if best_model is not None else combined_model
    if current_model is None:
        print("No valid models loaded. Will need to train new models.")

def train_model():
    global is_training, best_model, combined_model, current_model, preprocessor

    if is_training:
        return  # Exit if already training

    is_training = True
    print("Starting model training...")

    try:
        # Load dataset
        df = pd.read_csv('preprocessed_customer_churn_data.csv')

        # Drop the 'customerID' column as it is not needed for training
        df.drop(columns=['customerID'], inplace=True)

        # Handle 'TotalCharges' column separately
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

        # Ensure the correct data type for 'SeniorCitizen'
        df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)

        # Define feature columns and target column
        X = df.drop('Churn', axis=1)
        y = df['Churn']

        # Convert the target variable to numeric
        y = y.map({'Yes': 1, 'No': 0})

        print("Original class distribution:", Counter(y))

        # Preprocess the data
        X_preprocessed = preprocessor.fit_transform(X)

        # Apply SMOTEENN
        smoteenn = SMOTEENN(random_state=42)
        X_resampled, y_resampled = smoteenn.fit_resample(X_preprocessed, y)

        print("Resampled class distribution:", Counter(y_resampled))

        # Split the resampled data
        x_train, x_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

        # Initialize and train models
        models = {
            "Random Forest": RandomForestClassifier(random_state=42),
            "Bagging": BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, max_samples=0.25, bootstrap=False, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "Logistic Regression": LogisticRegression(random_state=42),
            "XGBoost": XGBClassifier(random_state=42)
        }

        # Train models and get predictions
        predictions = {}
        for name, model in models.items():
            model.fit(x_train, y_train)
            predictions[name] = model.predict(x_test)

        # Calculate accuracies
        accuracies = {name: accuracy_score(y_test, pred) for name, pred in predictions.items()}

        # Combine predictions
        all_preds = np.array(list(predictions.values())).T
        combined_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=all_preds)
        accuracies["Combined Model"] = accuracy_score(y_test, combined_pred)

        # Print individual model accuracies
        for name, acc in accuracies.items():
            print(f"{name} Accuracy: {acc:.4f}")

        # Find the best model
        best_model_name = max(accuracies, key=accuracies.get)
        print(f"\nThe best performing model is: {best_model_name} with an accuracy of {accuracies[best_model_name]:.4f}")

        # Save the best individual model with preprocessor
        if best_model_name != "Combined Model":
            best_classifier = models[best_model_name]
        else:
            best_classifier = models[max(models, key=lambda k: accuracies[k])]

        best_model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', best_classifier)])
        joblib.dump(best_model_pipeline, best_model_path)
        print(f"Best model saved successfully. Type: {type(best_model_pipeline)}")

        # Save the combined model pipeline
        combined_model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', BaggingClassifier(base_estimator=best_classifier, n_estimators=100, random_state=42))])
        joblib.dump(combined_model_pipeline, combined_model_path)
        print(f"Combined model saved successfully. Type: {type(combined_model_pipeline)}")

        best_model = best_model_pipeline
        combined_model = combined_model_pipeline
        current_model = best_model_pipeline

        print("Model training completed successfully.")
    except Exception as e:
        print(f"An error occurred during model training: {str(e)}")
    finally:
        is_training = False

# Scheduler to retrain model every 2 hours
scheduler = BackgroundScheduler()
scheduler.add_job(train_model, 'interval', hours=2)
scheduler.start()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    global best_model, combined_model, current_model

    print("Prediction route called")
    print(f"Current model type: {type(current_model)}")

    if is_training:
        return jsonify({"error": "Model is currently retraining. Please try again later."})

    if current_model is None:
        return jsonify({"error": "No model is currently available for predictions. Please try again later."})

    try:
        data = {
            'tenure': int(request.form.get('tenure')),
            'MonthlyCharges': float(request.form.get('MonthlyCharges')),
            'TotalCharges': float(request.form.get('TotalCharges')),
            'gender': request.form.get('gender'),
            'SeniorCitizen': int(request.form.get('SeniorCitizen')),
            'Partner': request.form.get('Partner'),
            'Dependents': request.form.get('Dependents'),
            'PhoneService': request.form.get('PhoneService'),
            'MultipleLines': request.form.get('MultipleLines'),
            'InternetService': request.form.get('InternetService'),
            'OnlineSecurity': request.form.get('OnlineSecurity'),
            'OnlineBackup': request.form.get('OnlineBackup'),
            'DeviceProtection': request.form.get('DeviceProtection'),
            'TechSupport': request.form.get('TechSupport'),
            'StreamingTV': request.form.get('StreamingTV'),
            'StreamingMovies': request.form.get('StreamingMovies'),
            'Contract': request.form.get('Contract'),
            'PaperlessBilling': request.form.get('PaperlessBilling'),
            'PaymentMethod': request.form.get('PaymentMethod')
        }

        df = pd.DataFrame([data])
        
        model_choice = request.form.get('model_choice')
        if model_choice == 'best_model':
            if best_model is None:
                return jsonify({"error": "Best model is not available. Please try the combined model."})
            print("Using best model")
            prediction = best_model.predict(df)
            probability = best_model.predict_proba(df)[:, 1][0]
            model = best_model
        else:
            if combined_model is None:
                return jsonify({"error": "Combined model is not available. Please try the best model."})
            print("Using combined model")
            prediction = combined_model.predict(df)
            probability = combined_model.predict_proba(df)[:, 1][0]
            model = combined_model
        
        # Calculate feature importance
        feature_importance = {}
        if hasattr(model[-1], 'feature_importances_'):
            importances = model[-1].feature_importances_
            feature_names = model[0].get_feature_names_out()
            feature_importance = dict(zip(feature_names, importances))
            feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5])
        
        return jsonify({
            "prediction": "Yes" if prediction[0] == 1 else "No",
            "probability": probability,
            "feature_importance": feature_importance
        })
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"})

@app.route('/train', methods=['POST'])
def train():
    global is_training
    if not is_training:
        threading.Thread(target=train_model).start()
        return jsonify({"message": "Model training started"})
    else:
        return jsonify({"message": "Model is already training"})

if __name__ == "__main__":
    load_models()  # Load existing models if available
    if current_model is None:
        train_model()  # Train the model if no models are available
    app.run(debug=True)