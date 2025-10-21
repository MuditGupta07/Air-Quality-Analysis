import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import json
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Urban Air Quality Explorer",
    page_icon="💨",
    layout="wide"
)

# --- Helper Functions ---
@st.cache_data
def load_data(path):
    """Loads the air quality data from a CSV file."""
    try:
        df = pd.read_csv(path)
        # Basic cleaning: drop rows where essential columns are missing
        df.dropna(subset=['pm25_ugm3', 'no2_ugm3', 'year', 'city', 'country'], inplace=True)
        df['year'] = df['year'].astype(int)
        return df
    except FileNotFoundError:
        st.error(f"Error: The data file was not found at {path}. Please make sure 'air_quality_global.csv' is in the 'data/' directory.")
        return None

@st.cache_resource
def load_model(path):
    """Loads a pre-trained model."""
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        st.error(f"Error: The model file was not found at {path}. Please run the `train_model.py` script first.")
        return None

@st.cache_data
def load_metadata(path):
    """Loads metadata from a JSON file."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning(f"Metadata file not found at {path}.")
        return None

# --- Data and Model Loading ---
DATA_PATH = 'data/air_quality_global.csv'
MODEL_PATH = 'models/pm25_predictor.joblib'
METADATA_PATH = 'data/metadata.json'

data = load_data(DATA_PATH)
model = load_model(MODEL_PATH)
metadata = load_metadata(METADATA_PATH)

# --- App Title ---
st.title("💨 Urban Air Quality Interactive Explorer")
st.markdown("An application to explore, visualize, and predict urban air quality metrics based on the Global Urban Air Quality Dataset.")

if data is not None:
    # --- Sidebar for Filters ---
    st.sidebar.header("Data Filters")
    
    # Country Filter
    countries = sorted(data['country'].unique())
    selected_country = st.sidebar.selectbox("Select a Country", ["All"] + countries)

    # City Filter
    if selected_country != "All":
        cities = sorted(data[data['country'] == selected_country]['city'].unique())
        selected_city = st.sidebar.selectbox("Select a City", ["All"] + cities)
    else:
        selected_city = "All"

    # Year Filter
    min_year, max_year = int(data['year'].min()), int(data['year'].max())
    selected_year_range = st.sidebar.slider("Select Year Range", min_year, max_year, (min_year, max_year))

    # Apply filters
    filtered_data = data[
        (data['year'] >= selected_year_range[0]) & (data['year'] <= selected_year_range[1])
    ]
    if selected_country != "All":
        filtered_data = filtered_data[filtered_data['country'] == selected_country]
    if selected_city != "All":
        filtered_data = filtered_data[filtered_data['city'] == selected_city]

    # --- Main Application Tabs ---
    tab1, tab2, tab3 = st.tabs(["🌎 Exploratory Data Analysis (EDA)", "🤖 Predictive Model", "ℹ️ About & Metadata"])

    # --- TAB 1: EDA ---
    with tab1:
        st.header("Exploratory Data Analysis (EDA)")
        
        if filtered_data.empty:
            st.warning("No data available for the selected filters. Please adjust your selection.")
        else:
            # Key Metrics
            col1, col2, col3 = st.columns(3)
            avg_pm25 = filtered_data['pm25_ugm3'].mean()
            avg_no2 = filtered_data['no2_ugm3'].mean()
            records = len(filtered_data)
            col1.metric("Total Records", f"{records:,}")
            col2.metric("Average PM2.5 (µg/m³)", f"{avg_pm25:.2f}")
            col3.metric("Average NO₂ (µg/m³)", f"{avg_no2:.2f}")

            st.markdown("---")
            
            # Interactive Visualizations
            st.subheader("Time Series of Pollutants")
            time_series_df = filtered_data.groupby('year')[['pm25_ugm3', 'no2_ugm3']].mean().reset_index()
            fig_ts = px.line(time_series_df, x='year', y=['pm25_ugm3', 'no2_ugm3'], title="Average Pollutant Levels Over Time", labels={'value': 'Concentration (µg/m³)'})
            st.plotly_chart(fig_ts, use_container_width=True)

            st.subheader("Distribution of PM2.5 Levels")
            fig_hist = px.histogram(filtered_data, x='pm25_ugm3', nbins=50, title="Histogram of PM2.5 Concentrations")
            st.plotly_chart(fig_hist, use_container_width=True)

            st.subheader("Geospatial Distribution of PM2.5")
            map_data = filtered_data.groupby(['city', 'country', 'latitude', 'longitude'])['pm25_ugm3'].mean().reset_index()
            fig_map = px.scatter_geo(
                map_data, 
                lat='latitude', 
                lon='longitude', 
                size='pm25_ugm3',
                color='pm25_ugm3',
                hover_name='city',
                projection='natural earth',
                title='Average PM2.5 Concentration by City',
                color_continuous_scale=px.colors.sequential.YlOrRd
            )
            st.plotly_chart(fig_map, use_container_width=True)

    # --- TAB 2: Predictive Model ---
    with tab2:
        st.header("🤖 PM2.5 Prediction Model")
        
        if model is None:
            st.error("Model could not be loaded. Please ensure `pm25_predictor.joblib` exists in the `models/` directory.")
        else:
            st.markdown("### Make a Custom Prediction")
            st.write("Input values below to predict the PM2.5 concentration.")
            
            # Input features
            col1, col2 = st.columns(2)
            with col1:
                year_input = st.number_input("Year", min_value=2000, max_value=2030, value=2025)
                month_input = st.slider("Month", 1, 12, 6)
                lat_input = st.number_input("Latitude", value=40.71, format="%.4f")
            with col2:
                lon_input = st.number_input("Longitude", value=-74.00, format="%.4f")
                no2_input = st.number_input("NO₂ Concentration (µg/m³)", value=30.0, format="%.2f")

            if st.button("Predict PM2.5"):
                features = np.array([[lat_input, lon_input, year_input, month_input, no2_input]])
                prediction = model.predict(features)
                st.success(f"**Predicted PM2.5 Concentration:** `{prediction[0]:.2f} µg/m³`")
            
            st.markdown("---")
            st.markdown("### Model Interpretation & Evaluation")
            st.write("""
            The model uses a RandomForest Regressor, which is an ensemble of decision trees. 
            It makes predictions by averaging the output of many individual trees.
            """)

            # Feature Importance
            if hasattr(model, 'feature_importances_'):
                st.subheader("Feature Importances")
                st.write("This chart shows which factors the model considers most important when making a prediction.")
                feature_names = ['latitude', 'longitude', 'year', 'month', 'no2_ugm3']
                importances = pd.DataFrame({'feature': feature_names, 'importance': model.feature_importances_})
                importances = importances.sort_values('importance', ascending=False)
                
                fig_imp = px.bar(importances, x='importance', y='feature', orientation='h', title="Model Feature Importances")
                st.plotly_chart(fig_imp, use_container_width=True)

    # --- TAB 3: About & Metadata ---
    with tab3:
        st.header("ℹ️ About the Project & Dataset")
        
        st.subheader("Project Goal")
        st.write("""
        This application was created to fulfill the requirements of the Data Science Assessment. 
        It provides an interactive interface to explore the Global Urban Air Quality dataset, 
        visualize trends, and interact with a predictive machine learning model.
        """)

        st.subheader("Dataset Metadata")
        if metadata:
            st.json(metadata)
        else:
            st.warning("Could not load metadata.json.")

else:
    st.error("Application cannot start. Please check the data loading errors above.")
