import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import joblib
import os

print("Starting model training process...")

# --- 1. Load Data ---
DATA_PATH = 'data/air_quality_global.csv'
try:
    df = pd.read_csv(DATA_PATH)
    print(f"Successfully loaded data from {DATA_PATH}. Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: The data file was not found at {DATA_PATH}")
    print("Please make sure 'air_quality_global.csv' is in the 'data/' directory.")
    exit()

# --- 2. Data Preprocessing ---
print("Performing data preprocessing...")
# Select features and target
features = ['latitude', 'longitude', 'year', 'month', 'no2_ugm3']
target = 'pm25_ugm3'

df_model = df[features + [target]].copy()

# Handle missing values based on a simple strategy (e.g., median imputation)
# This aligns with the 'missing-value strategy' to be documented in README [cite: 55]
imputer = SimpleImputer(strategy='median')
df_model_imputed = pd.DataFrame(imputer.fit_transform(df_model), columns=df_model.columns)
print("Missing values handled using median imputation.")

X = df_model_imputed[features]
y = df_model_imputed[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # [cite: 72]
print(f"Data split into training ({len(X_train)} rows) and testing ({len(X_test)} rows) sets.")

# --- 3. Modeling and Hyperparameter Tuning ---
print("Training RandomForestRegressor model with GridSearchCV...")
# Define the parameter grid for GridSearchCV [cite: 32]
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20],
    'min_samples_leaf': [5, 10]
}

# Initialize the model and the grid search
rf = RandomForestRegressor(random_state=42) # [cite: 72]
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1, scoring='r2')

# Fit the model
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"GridSearchCV completed. Best parameters found: {grid_search.best_params_}")

# --- 4. Model Evaluation ---
print("Evaluating the best model...")
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance on Test Set:")
print(f"  Mean Squared Error (MSE): {mse:.4f}")
print(f"  R-squared (R2 Score): {r2:.4f}")

# --- 5. Save the Model ---
# Ensure the models directory exists
os.makedirs('models', exist_ok=True)
MODEL_PATH = 'models/pm25_predictor.joblib'
joblib.dump(best_model, MODEL_PATH)
print(f"Model successfully saved to {MODEL_PATH}")

print("Training process finished.")