import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import date

# 1. DEFINE CUSTOM FUNCTION (Required for joblib to load your pipeline properly)
def littleTransform(df):
    Data = df.copy()
    
    # Is weekend
    Data['Is_Weekend'] = Data['Days'].apply(lambda x: 1 if x > 5 else 0)
    
    # Season
    Winter = [11, 12, 1, 2]
    Summer = [3, 4, 5, 6]
    Monsoon = [7, 8, 9, 10]
    
    def assign_season(month):
        if month in Winter: return 'Winter'
        elif month in Summer: return 'Summer'
        else: return 'Monsoon'
        
    Data['season'] = Data['Month'].apply(assign_season)
    Data['season'] = Data['season'].astype('category')
    
    # Engineered Features
    Data['coarse_particles'] = Data['PM10'] - Data['PM2.5']
    Data['traffic_aerosol_index'] = Data['PM2.5'] / Data['NO2']
    Data['Secondary_Particle_Proxy'] = Data['PM2.5'] / Data['CO']
    Data['Total_Particulate_Load'] = Data['PM2.5'] + Data['PM10']
    Data['Dominant_Particle_Share'] = Data['PM2.5'] / Data['Total_Particulate_Load']
    Data['Combustion_Proxy'] = Data['PM2.5'] * Data['NO2'] * Data['CO']
    Data['Acid_Rain'] = Data['SO2'] * Data['NO2']
    Data['PM2.5/PM10'] = Data['PM2.5'] / Data['PM10']
    
    return Data

# 2. LOAD MODELS (Cached so it runs fast)
@st.cache_resource
def load_models():
    pipeline = joblib.load('my_pipeline.pkl')
    model = joblib.load('rf_model.pkl')
    return pipeline, model

try:
    pipeline, model = load_models()
except FileNotFoundError:
    st.error("Model files not found! Please ensure 'my_pipeline.pkl' and 'rf_model.pkl' are in the same folder.")

# 3. APP UI
st.title("🌍 Tomorrow's AQI Forecaster")
st.write("Enter today's weather and pollution data to predict tomorrow's Air Quality Index.")

today = date.today()

# 4. WRAP INPUTS IN A FORM TO PREVENT LAG / RE-RUNS ON EVERY KEYPRESS
with st.form("aqi_form"):
    st.header("Today's Data")
    col1, col2 = st.columns(2)

    with col1:
        date_input = st.number_input("Date", min_value=1, max_value=31, value=today.day)
        month = st.number_input("Month", min_value=1, max_value=12, value=today.month)
        year = st.number_input("Year", min_value=2021, max_value=2050, value=today.year)
        days = st.number_input("Day of Week (1=Mon, 7=Sun)", min_value=1, max_value=7, value=today.isoweekday())
        holidays = st.selectbox("Is today a Holiday?", [0, 1])

    with col2:
        pm25 = st.number_input("PM2.5", min_value=0.1, value=50.0)
        pm10 = st.number_input("PM10", min_value=0.1, value=100.0)
        no2 = st.number_input("NO2", min_value=0.1, value=30.0)
        so2 = st.number_input("SO2", min_value=0.1, value=15.0)
        co = st.number_input("CO", min_value=0.1, value=1.0)
        ozone = st.number_input("Ozone", min_value=0.1, value=35.0)

    # Form Submit Button
    submitted = st.form_submit_button("🔮 Predict Tomorrow's AQI")

# 5. PREDICTION LOGIC
if submitted:
    input_data = pd.DataFrame({
        'Date': [date_input],
        'Month': [month],
        'Year': [year],
        'Holidays_Count': [holidays],
        'Days': [days],
        'PM2.5': [pm25],
        'PM10': [pm10],
        'NO2': [no2],
        'SO2': [so2],
        'CO': [co],
        'Ozone': [ozone]
    })
    
    try:
        # Push through the custom pipeline
        transformed_data = pipeline.transform(input_data)
        
        # Predict using Random Forest
        prediction = model.predict(transformed_data)[0]
        
        st.success(f"### Predicted AQI for Tomorrow: {int(prediction)}")
        
        # Display context based on US EPA AQI guidelines
        if prediction <= 50:
            st.info("🟢 **Good:** Air quality is satisfactory, and air pollution poses little or no risk.")
        elif prediction <= 100:
            st.warning("🟡 **Moderate:** Air quality is acceptable. However, there may be a risk for some people.")
        elif prediction <= 150:
            st.warning("🟠 **Unhealthy for Sensitive Groups:** Members of sensitive groups may experience health effects.")
        elif prediction <= 200:
            st.error("🔴 **Unhealthy:** Some members of the general public may experience health effects.")
        else:
            st.error("🟣 **Very Unhealthy / Hazardous:** Health warnings of emergency conditions.")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
