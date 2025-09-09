End-to-End Customer Churn Prediction Pipeline
This project demonstrates how to build a complete and reusable machine learning pipeline for predicting customer churn using the Telco Churn dataset. It leverages Scikit-learn's Pipeline and GridSearchCV to streamline preprocessing, model training, and hyperparameter tuning.

Project Objective
The goal is to create a production-ready workflow that encapsulates all steps of the machine learning process, from raw data to a trained, optimized, and exportable model. This ensures consistency, reusability, and easier deployment.

Features
Data Loading and Cleaning: Loads the Telco Churn dataset and handles missing values.

Preprocessing: Uses ColumnTransformer to apply standard scaling to numerical features and one-hot encoding to categorical features.

ML Pipelines: Implements two separate pipelines for Logistic Regression and Random Forest classifiers.

Hyperparameter Tuning: Utilizes GridSearchCV to find the optimal hyperparameters for both models based on cross-validation accuracy.

Model Evaluation: Compares the tuned models and selects the best performer.

Model Export: Saves the entire, fitted pipeline (including preprocessors and the trained model) to a single file using joblib for easy reuse.

Skills Gained
Building robust ML pipelines with Scikit-learn.

Performing automated hyperparameter tuning with GridSearchCV.

Understanding production-ready practices like model serialization.

Creating a reusable and modular machine learning workflow.

Setup and Installation
Clone the repository or download the files.

Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install the required dependencies:

pip install -r requirements.txt

How to Run
Execute the main Python script from your terminal:

python churn_pipeline.py

What the Script Does:
Loads Data: Fetches the Telco Churn dataset from a public URL.

Prepares Data: Cleans the data and splits it into training and testing sets.

Builds Pipelines: Defines the preprocessing steps for different data types.

Trains Models: Runs GridSearchCV to train and tune both a Logistic Regression and a Random Forest model. This may take a few moments.

Evaluates and Selects: Prints the best parameters and cross-validation scores for each model, then selects the overall best model.

Exports Pipeline: Saves the best-performing pipeline to a file named churn_prediction_pipeline.joblib. This file contains everything needed to make predictions on new, raw data.

After running, you will have the churn_prediction_pipeline.joblib file in your directory, ready for deployment or integration into another application.