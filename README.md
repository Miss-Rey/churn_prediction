# Customer Churn Prediction

## Project Overview

This project aims to predict customer churn using machine learning models. Customer churn occurs when customers stop doing business with a company. Understanding and predicting churn can help businesses take proactive measures to retain customers. This project involves data analysis, feature engineering, and model building to create a reliable churn prediction system.

## Table of Contents

- [Customer Churn Prediction](#customer-churn-prediction)
  - [Project Overview](#project-overview)
  - [Table of Contents](#table-of-contents)
  - [Dataset](#dataset)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Model Training and Evaluation](#model-training-and-evaluation)
  - [Results](#results)
  - [Contributing](#contributing)
  - [License](#license)

## Dataset

The dataset used for this project is sourced from Kaggle and contains information about customer demographics, account information, and service usage.

**Dataset features:**

- `customerID`: Unique identifier for each customer
- `gender`: Gender of the customer (Male/Female)
- `SeniorCitizen`: Indicates if the customer is a senior citizen (1/0)
- `Partner`: Indicates if the customer has a partner (Yes/No)
- `Dependents`: Indicates if the customer has dependents (Yes/No)
- `tenure`: Number of months the customer has stayed with the company
- `PhoneService`: Indicates if the customer has phone service (Yes/No)
- `MultipleLines`: Indicates if the customer has multiple lines (Yes/No/No phone service)
- `InternetService`: Type of internet service (DSL/Fiber optic/No)
- `OnlineSecurity`: Indicates if the customer has online security (Yes/No/No internet service)
- `OnlineBackup`: Indicates if the customer has online backup (Yes/No/No internet service)
- `DeviceProtection`: Indicates if the customer has device protection (Yes/No/No internet service)
- `TechSupport`: Indicates if the customer has tech support (Yes/No/No internet service)
- `StreamingTV`: Indicates if the customer has streaming TV (Yes/No/No internet service)
- `StreamingMovies`: Indicates if the customer has streaming movies (Yes/No/No internet service)
- `Contract`: Type of contract (Month-to-month/One year/Two year)
- `PaperlessBilling`: Indicates if the customer has paperless billing (Yes/No)
- `PaymentMethod`: Payment method (Electronic check/Mailed check/Bank transfer (automatic)/Credit card (automatic))
- `MonthlyCharges`: The amount charged to the customer monthly
- `TotalCharges`: The total amount charged to the customer
- `Churn`: Indicates if the customer churned (Yes/No)

## Project Structure

Customer-Churn-Prediction/
├── app.py # Flask web application for model serving
├── Customer Churn Analysis.ipynb # Data analysis and exploration notebook
├── Customer Churn Model Building.ipynb # Model building and training notebook
├── best_individual_model.pkl # Serialized best individual model
├── combined_model.pkl # Serialized combined model
└── README.md # Project documentation

## Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/yourusername/Customer-Churn-Prediction.git
   cd Customer-Churn-Prediction
   ```

2. Set up a virtual environment:

   ```sh
   python -m venv venv
   ```

3. Activate the virtual environment:

   - On Windows:
     ```sh
     venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```sh
     source venv/bin/activate
     ```

4. Install the required packages:

   ```sh
   pip install -r requirements.txt
   ```

5. Run the Flask application:

   ```sh
   python app.py
   ```

6. Access the web application:
   Open your web browser and navigate to `http://localhost:5000`.

## Usage

You can also access the hosted version of the application at:

[Customer Churning Prediction](https://customerchurningprediction.azurewebsites.net)

## Exploratory Data Analysis (EDA)

The Exploratory Data Analysis (EDA) is conducted in the `Customer Churn Analysis.ipynb` notebook. This notebook includes:

- Loading and displaying the dataset.
- Checking for missing values.
- Visualizing the distribution of key features.
- Analyzing the correlation between features and churn.
- Identifying important features that contribute to customer churn.

## Model Training and Evaluation

The model training and evaluation process is covered in the `Customer Churn Model Building.ipynb` notebook. The steps include:

1. **Data Preprocessing:**

   - Handling missing values
   - Encoding categorical variables
   - Scaling numerical features

2. **Model Building:**

   - Implementing various machine learning models such as RandomForest, GradientBoosting, XGBoost, and Logistic Regression.
   - Using ensemble methods to improve model performance.

3. **Model Evaluation:**
   - Assessing model performance using metrics such as accuracy, precision, recall, and F1-score.
   - Selecting the best model based on evaluation metrics.

## Results

The project resulted in a highly accurate churn prediction model. The feature importance analysis provided insights into the factors contributing to customer churn. The web application allows users to input customer data and receive real-time predictions along with feature importance.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or additions.

## License

This project is not licensed.
