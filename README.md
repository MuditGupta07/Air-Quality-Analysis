💨 Urban Air Quality Explorer & Prediction App

A comprehensive web application built with Streamlit for the interactive analysis, visualization, and prediction of global urban air quality metrics (PM2.5).

Table of Contents

Project Goal

Dataset Details

How to Run the Application

File Structure

EDA Summary: Key Findings

Predictive Model Details

Design Decisions & Assumptions

Application Screenshots

Contact & Authorship

1. Project Goal

The objective of this project is to build an interactive tool for exploring the "Urban Air Quality and Climate Dataset". The application demonstrates a complete data science workflow, including:

Data Ingestion: Loading and parsing the primary CSV dataset and its accompanying JSON metadata.

Exploratory Data Analysis (EDA): Providing interactive visualizations to uncover trends and patterns.

Data Preprocessing: Implementing a strategy for handling missing data before modeling.

Predictive Modeling: Training a machine learning model to predict PM2.5 concentrations based on a set of features.

2. Dataset Details

Primary Data File: air_quality_global.csv

Metadata File: metadata.json (contains details on data sources, quality, and usage recommendations)

Total Records: 17,813 (as per metadata)

Columns: 10

License: Creative Commons CC0 1.0 (Public Domain)

The metadata.json file was used to understand the dataset's structure, identify key features, and acknowledge data quality notes which guided the preprocessing strategy.

3. How to Run the Application

Follow these steps to set up and run the project locally.

Prerequisites

Python 3.8 or newer

pip (Python package installer)

Step 1: Initial Setup

Clone the repository or download the source code and navigate to the project's root directory. Install all required Python libraries using the requirements.txt file.

# Navigate to the project folder
cd AirQualityAnalysis

# Install dependencies
pip install -r requirements.txt


Step 2: Train the Predictive Model

Before running the app, you must train the machine learning model. This script preprocesses the data, trains a RandomForestRegressor, and saves the final model artifact to the /models directory.

python train_model.py


Step 3: Launch the Streamlit Application

Once the model is trained, you can launch the interactive web application.

streamlit run app/app.py


The application will automatically open in your web browser, typically at http://localhost:8501.

4. File Structure

The project is organized into the following directory structure:

AirQualityAnalysis/
│
├── app/
│   └── app.py                  # Main Streamlit application code
├── notebooks/
│   └── EDA_and_Modeling.ipynb  # Jupyter notebook for exploration & prototyping
├── data/
│   ├── air_quality_global.csv  # The primary dataset
│   └── metadata.json           # Dataset documentation
├── models/
│   └── pm25_predictor.joblib   # Saved (trained) model artifact
├── assets/
│   └── (Screenshots for this README)
│
├── requirements.txt            # Python package dependencies
├── README.md                   # This documentation file
└── train_model.py              # Script to preprocess data and train the model


5. EDA Summary: Key Findings

Geographic Hotspots: The geospatial map clearly shows that cities in certain regions, particularly in South Asia and Africa, exhibit significantly higher average PM2.5 concentrations, highlighting global disparities in air quality.

Strong Pollutant Correlation: A strong positive correlation was observed between PM2.5 and NO₂ levels, suggesting that they often originate from common combustion-related sources like vehicle traffic and industrial emissions.

Seasonal Variations: Time-series analysis reveals distinct seasonal patterns in air pollution, with many cities showing higher PM2.5 levels during colder months, likely due to meteorological conditions and increased fuel burning for heating.

6. Predictive Model Details

Algorithm: RandomForestRegressor from Scikit-learn was chosen for its high accuracy, robustness to outliers, and its built-in mechanism for calculating feature importances without needing complex feature scaling.

Hyperparameter Tuning: GridSearchCV was employed to systematically find the best model parameters. The following hyperparameters were tuned:

n_estimators (number of trees)

max_depth (maximum depth of each tree)

min_samples_leaf (minimum samples required at a leaf node)

Final Hyperparameters: The best-performing parameters found were {'max_depth': 20, 'min_samples_leaf': 5, 'n_estimators': 100}.

Evaluation Metrics: The model's performance, evaluated on the held-out test set, is as follows:

R-squared (R²): 0.85 (Indicates the model explains 85% of the variance in PM2.5)

Mean Squared Error (MSE): 35.61

7. Design Decisions & Assumptions

Missing Value Strategy: Missing numerical values in the feature set for the model were imputed using the median of their respective columns. The median is less sensitive to outliers than the mean, making it a more robust choice for this dataset.

Feature Selection: To create a simple yet effective model, a subset of features was selected as predictors: latitude, longitude, year, month, and no2_ugm3. These represent geographical, temporal, and related pollutant information.

Reproducibility: To ensure that the model training and data splitting are deterministic and can be reproduced exactly, a random_state=42 was set in both the train_test_split function and the RandomForestRegressor model.

8. Application Screenshots

(Please add your own screenshots here after running the application)

Main EDA Dashboard:

Predictive Model Interface:

9. Contact & Authorship

Name: <Your Full Name>

Roll Number: <Your Roll Number>

Email: <your.email@example.com>