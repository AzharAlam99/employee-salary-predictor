import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os

def train_and_save_model():
    """
    Loads, preprocesses, trains, evaluates, and saves the salary prediction model.
    This function tests multiple models and selects the best one for deployment.
    """
    # --- 1. Data Loading and Cleaning ---
    try:
        df = pd.read_csv('indian_employee_salary_dataset.csv')
        print("✅ Dataset loaded successfully.")
    except FileNotFoundError:
        print("❌ Error: 'indian_employee_salary_dataset.csv' not found. Make sure it's in the project folder.")
        return

    # Drop rows with any missing values for a clean model
    df.dropna(inplace=True)
    # Rename columns for easier access and clarity
    df.rename(columns={'Experience (Years)': 'Experience', 'Monthly Salary (INR)': 'Salary'}, inplace=True)
    
    # --- 2. Feature Engineering ---
    # Define the features (X) that will be used for prediction and the target (y) variable
    X = df[['Age', 'Gender', 'Education', 'Job Title', 'Experience']]
    y = df['Salary']

    # --- 3. Preprocessing ---
    # Identify categorical and numerical features to apply different transformations
    categorical_features = ['Gender', 'Education', 'Job Title']
    numerical_features = ['Age', 'Experience']

    # Create preprocessing pipelines for numerical and categorical features
    # StandardScaler standardizes numerical features (mean=0, variance=1)
    numerical_transformer = StandardScaler()
    # OneHotEncoder converts categorical text data into a machine-readable format
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # Create a preprocessor object using ColumnTransformer to apply specific transformers to specific columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Keep other columns (if any) untouched
    )

    # --- 4. Model Training and Selection ---
    # Define the models to be tested
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror', n_jobs=-1)
    }

    best_model_name = ''
    best_model_score = -np.inf
    best_model_pipeline = None

    # Split data into training and testing sets for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n--- Model Evaluation ---")
    for name, model in models.items():
        # Create the full pipeline: first preprocess the data, then train the model
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Make predictions on the test set
        y_pred = pipeline.predict(X_test)
        
        # Evaluate the model using R² score and RMSE
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"--- {name} ---")
        print(f"  R² Score: {r2:.4f}")
        print(f"  RMSE: ₹{rmse:,.2f}")
        
        # Select the best model based on the R² score
        if r2 > best_model_score:
            best_model_score = r2
            best_model_name = name
            best_model_pipeline = pipeline

    print(f"\n🏆 Best Model Selected: {best_model_name} with R² score of {best_model_score:.4f}")

    # --- 5. Saving Artifacts ---
    # Create the 'model' directory if it doesn't exist
    os.makedirs("model", exist_ok=True)
    
    # Save the complete trained pipeline (preprocessor + best model)
    joblib.dump(best_model_pipeline, 'model/salary_prediction_model.joblib')
    print("\n✅ Trained model pipeline saved to 'model/salary_prediction_model.joblib'")
    
    # Save the feature columns in the correct order for the web app
    feature_columns = list(X.columns)
    joblib.dump(feature_columns, 'model/feature_columns.joblib')
    print("✅ Feature columns saved to 'model/feature_columns.joblib'")
    
    # Save the unique values from categorical columns for the dropdown menus in the web app
    categorical_values = {col: X[col].unique().tolist() for col in categorical_features}
    joblib.dump(categorical_values, 'model/categorical_values.joblib')
    print("✅ Categorical values for dropdowns saved to 'model/categorical_values.joblib'")

if __name__ == '__main__':
    train_and_save_model()
