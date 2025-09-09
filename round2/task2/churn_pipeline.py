# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import numpy as np

# --- 1. Load and Prepare Data ---
print("Loading data...")
# Load the dataset from a public URL
url = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'
df = pd.read_csv(url)

# Drop customerID as it's not a predictive feature
df = df.drop('customerID', axis=1)

# Convert TotalCharges to numeric, coercing errors to NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Impute missing values in TotalCharges with the column median
# This happens for new customers who haven't been charged yet
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Convert target variable 'Churn' to binary
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

print("Data loaded and pre-processed successfully.")
print(df.head())

# --- 2. Identify Feature Types and Split Data ---
# Separate target variable from features
X = df.drop('Churn', axis=1)
y = df['Churn']

# Identify numerical and categorical features
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include=object).columns.tolist()

print(f"\nNumerical features: {numerical_features}")
print(f"Categorical features: {categorical_features}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")


# --- 3. Define Preprocessing Steps ---
# Create a preprocessor for numerical and categorical features
# Numerical features will be scaled.
# Categorical features will be one-hot encoded.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough' # Keep other columns (if any), though we've defined all
)


# --- 4. Create Full ML Pipelines with GridSearchCV ---
print("\nSetting up pipelines and GridSearch...")

# --- Pipeline 1: Logistic Regression ---
pipeline_lr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

# Define hyperparameters for GridSearchCV for Logistic Regression
param_grid_lr = {
    'classifier__C': [0.1, 1.0, 10],
    'classifier__solver': ['liblinear', 'saga']
}

# Create GridSearchCV object for Logistic Regression
grid_search_lr = GridSearchCV(pipeline_lr, param_grid_lr, cv=5, n_jobs=-1, scoring='accuracy', verbose=1)


# --- Pipeline 2: Random Forest ---
pipeline_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Define hyperparameters for GridSearchCV for Random Forest
param_grid_rf = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# Create GridSearchCV object for Random Forest
grid_search_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=5, n_jobs=-1, scoring='accuracy', verbose=1)


# --- 5. Train Models ---
print("\n--- Training Logistic Regression ---")
grid_search_lr.fit(X_train, y_train)

print("\n--- Training Random Forest ---")
grid_search_rf.fit(X_train, y_train)


# --- 6. Evaluate Models and Select the Best ---
print("\n--- Model Evaluation ---")

# Logistic Regression Results
print("\nBest Logistic Regression Parameters:")
print(grid_search_lr.best_params_)
print(f"Best LR Cross-validation Accuracy: {grid_search_lr.best_score_:.4f}")

# Random Forest Results
print("\nBest Random Forest Parameters:")
print(grid_search_rf.best_params_)
print(f"Best RF Cross-validation Accuracy: {grid_search_rf.best_score_:.4f}")

# Compare models and select the best one
if grid_search_lr.best_score_ > grid_search_rf.best_score_:
    best_model = grid_search_lr.best_estimator_
    best_model_name = "Logistic Regression"
    best_score = grid_search_lr.best_score_
else:
    best_model = grid_search_rf.best_estimator_
    best_model_name = "Random Forest"
    best_score = grid_search_rf.best_score_

print(f"\nSelected Best Model: {best_model_name} with CV Accuracy: {best_score:.4f}")

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy of the best model: {test_accuracy:.4f}")
print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_pred))


# --- 7. Export the Final Pipeline ---
pipeline_filename = 'churn_prediction_pipeline.joblib'
joblib.dump(best_model, pipeline_filename)

print(f"\n✅ Complete pipeline successfully exported to '{pipeline_filename}'")

# --- Example of Loading and Using the Pipeline ---
print("\n--- Loading and testing the exported pipeline ---")
loaded_pipeline = joblib.load(pipeline_filename)

# Create a single example for prediction (as a DataFrame)
sample_data = X_test.iloc[0:1]
prediction = loaded_pipeline.predict(sample_data)
prediction_proba = loaded_pipeline.predict_proba(sample_data)

print(f"Prediction for sample data: {'Churn' if prediction[0] == 1 else 'No Churn'}")
print(f"Prediction probability: {prediction_proba}")
