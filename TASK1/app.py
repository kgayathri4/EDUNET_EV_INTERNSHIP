import streamlit as st
import pandas as pd
import joblib

st.title("🔋 Electric Vehicle Price Predictor")

# Load trained model
pipe = joblib.load('ev_price_pipeline.pkl')

st.subheader("Enter EV Specifications:")

battery = st.number_input('Battery (kWh)', 20.0, 150.0, 75.0)
efficiency = st.number_input('Efficiency (Wh/km)', 100, 300, 200)
fastcharge = st.number_input('Fast Charge (kW)', 30.0, 350.0, 150.0)
range_km = st.number_input('Range (km)', 100, 800, 400)
top_speed = st.number_input('Top Speed (km/h)', 100, 300, 200)
acceleration = st.number_input('0–100 km/h (seconds)', 2.0, 15.0, 6.0)

if st.button('Predict Price (€)'):
    new_data = pd.DataFrame([{
        'Battery': battery,
        'Efficiency': efficiency,
        'Fast_charge': fastcharge,
        'Range': range_km,
        'Top_speed': top_speed,
        'Acceleration_0_100': acceleration
    }])
    pred = pipe.predict(new_data)[0]
    st.success(f"💰 Predicted EV Price: €{pred:,.2f}")