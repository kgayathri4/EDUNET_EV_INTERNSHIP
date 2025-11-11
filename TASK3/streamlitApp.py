# ===============================
#  Unified Electric Vehicle Predictor
# ===============================

import streamlit as st
import pandas as pd
import joblib

# Set page config
st.set_page_config(page_title="EV Predictor", page_icon="🔋", layout="centered")

st.title("Electric Vehicle Prediction System")
st.markdown("### Choose between *Price Prediction* and *Range Prediction*")

# Sidebar menu
menu = st.sidebar.radio("Select Prediction Type", ["Price Prediction", "Range Prediction"])

# --------------------------
# PRICE PREDICTION SECTION
# --------------------------
if menu == "Price Prediction":
    st.header("Electric Vehicle Price Predictor")

    # Load price model
    price_model = joblib.load("ev_price_pipeline.pkl")

    st.subheader("Enter EV Specifications:")
    battery = st.number_input('Battery (kWh)', 20.0, 150.0, 75.0)
    efficiency = st.number_input('Efficiency (Wh/km)', 100, 300, 200)
    fastcharge = st.number_input('Fast Charge (kW)', 30.0, 350.0, 150.0)
    range_km = st.number_input('Range (km)', 100, 800, 400)
    top_speed = st.number_input('Top Speed (km/h)', 100, 300, 200)
    acceleration = st.number_input('0–100 km/h (seconds)', 2.0, 15.0, 6.0)

    if st.button('Predict EV Price (€)'):
        new_data = pd.DataFrame([{
            'Battery': battery,
            'Efficiency': efficiency,
            'Fast_charge': fastcharge,
            'Range': range_km,
            'Top_speed': top_speed,
            'Acceleration_0_100': acceleration
        }])

        pred = price_model.predict(new_data)[0]
        st.success(f" Predicted EV Price: €{pred:,.2f}")

# --------------------------
# RANGE PREDICTION SECTION
# --------------------------
elif menu == "Range Prediction":
    st.header(" Electric Vehicle Range Predictor")

    # Load range model
    range_model = joblib.load("ev_range_model.pkl")

    st.subheader("Enter EV Performance Specs:")
    top_speed = st.number_input('Top Speed (km/h)', 100, 400, 200)
    battery_capacity = st.number_input('Battery Capacity (kWh)', 20.0, 150.0, 75.0)
    torque = st.number_input('Torque (Nm)', 100, 1000, 400)

    if st.button('Predict EV Range (km)'):
        new_data = pd.DataFrame([{
            'top_speed_kmh': top_speed,
            'battery_capacity_kWh': battery_capacity,
            'torque_nm': torque
        }])

        pred = range_model.predict(new_data)[0]
        st.success(f" Predicted EV Range: {pred:.2f} km")

# Footer
st.markdown("---")
st.caption("Developed by Afnan Saif Afnan — Internship Final Combined Task")