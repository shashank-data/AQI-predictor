import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import date

# ---------------------------------------------------------
# 1. CUSTOM TRANSFORMER FUNCTION
# ---------------------------------------------------------
def littleTransform(df):
    Data = df.copy()
    
    # Is weekend calculation
    Data['Is_Weekend'] = Data['Days'].apply(lambda x: 1 if x > 5 else 0)
    
    # Season mapping
    Winter = [11, 12, 1, 2]
    Summer = [3, 4, 5, 6]
    Monsoon = [7, 8, 9, 10]
    
    def assign_season(month):
        if month in Winter: return 'Winter'
        elif month in Summer: return 'Summer'
        else: return 'Monsoon'
        
    Data['season'] = Data['Month'].apply(assign_season)
    Data['season'] = Data['season'].astype('category')
    
    # Feature Engineering (with tiny epsilon to prevent division by zero errors)
    eps = 1e-6
    Data['coarse_particles'] = Data['PM10'] - Data['PM2.5']
    Data['traffic_aerosol_index'] = Data['PM2.5'] / (Data['NO2'] + eps)
    Data['Secondary_Particle_Proxy'] = Data['PM2.5'] / (Data['CO'] + eps)
    Data['Total_Particulate_Load'] = Data['PM2.5'] + Data['PM10']
    Data['Dominant_Particle_Share'] = Data['PM2.5'] / (Data['Total_Particulate_Load'] + eps)
    Data['Combustion_Proxy'] = Data['PM2.5'] * Data['NO2'] * Data['CO']
    Data['Acid_Rain'] = Data['SO2'] * Data['NO2']
    Data['PM2.5/PM10'] = Data['PM2.5'] / (Data['PM10'] + eps)
    
    return Data

# ---------------------------------------------------------
# 2. LOAD MODELS SAFELY
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    try:
        pipeline = joblib.load('my_pipeline.pkl')
        model = joblib.load('rf_model.pkl')
        return pipeline, model, True
    except Exception:
        return None, None, False

pipeline, model, models_loaded = load_models()

# ---------------------------------------------------------
# 3. STREAMLIT USER INTERFACE
# ---------------------------------------------------------
st.title("🌍 Tomorrow's AQI Forecaster")
st.write("Enter today's weather and pollution metrics to estimate tomorrow's Air Quality Index.")

if not models_loaded:
    st.warning("⚠️ Model files (`my_pipeline.pkl` / `rf_model.pkl`) were not detected in the project folder. Please ensure both files are uploaded.")

today = date.today()

with st.form("aqi_form"):
    st.header("Today's Measurements")
    col1, col2 = st.columns(2)

    with col1:
        date_input = st.number_input("Date", min_value=1, max_value=31, value=today.day)
        month = st.number_input("Month", min_value=1, max_value=12, value=today.month)
        year = st.number_input("Year", min_value=2021, max_value=2050, value=today.year)
        days = st.number_input("Day of Week (1=Mon, 7=Sun)", min_value=1, max_value=7, value=today.isoweekday())
        holidays = st.selectbox("Is today a Holiday?", [0, 1])

    with col2:
        pm25 = st.number_input("PM2.5", min_value=0.0, value=50.0)
        pm10 = st.number_input("PM10", min_value=0.0, value=100.0)
        no2 = st.number_input("NO2", min_value=0.0, value=30.0)
        so2 = st.number_input("SO2", min_value=0.0, value=15.0)
        co = st.number_input("CO", min_value=0.0, value=1.0)
        ozone = st.number_input("Ozone", min_value=0.0, value=35.0)

    submitted = st.form_submit_button("🔮 Predict Tomorrow's AQI")

# ---------------------------------------------------------
# 4. PREDICTION EXECUTION
# ---------------------------------------------------------
if submitted:
    if not models_loaded:
        st.warning("Cannot run prediction until model files are present.")
    else:
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
            transformed_data = pipeline.transform(input_data)
            prediction = model.predict(transformed_data)[0]
            predicted_val = max(0, int(prediction))
            
            st.success(f"### Predicted AQI for Tomorrow: {predicted_val}")
            
            if predicted_val <= 50:
                st.info("🟢 **Good:** Air quality is satisfactory, and air pollution poses little or no risk.")
            elif predicted_val <= 100:
                st.warning("🟡 **Moderate:** Air quality is acceptable. However, there may be a risk for sensitive individuals.")
            elif predicted_val <= 150:
                st.warning("🟠 **Unhealthy for Sensitive Groups:** Members of sensitive groups may experience health effects.")
            elif predicted_val <= 200:
                st.error("🔴 **Unhealthy:** General public may begin to experience health effects.")
            else:
                st.error("🟣 **Very Unhealthy / Hazardous:** Health warnings of emergency conditions.")
                
        except Exception as e:
            st.warning(f"Unable to complete prediction with current inputs. (Details: {e})")
