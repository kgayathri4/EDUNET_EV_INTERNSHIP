# ===============================
#  Electric Vehicle Range Chatbot
# ===============================

import streamlit as st
import pandas as pd
import joblib
import re

# Load trained model
model = joblib.load('ev_range_model.pkl')

# Streamlit app
st.set_page_config(page_title="EV Range Chatbot", page_icon="🤖")
st.title(" Electric Vehicle Range Prediction Chatbot")

st.markdown("""
Chat with the bot to predict your electric vehicle's *range (km)*  
based on top speed, battery capacity, and torque.
""")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Type your EV details... (e.g., 'battery 80, speed 220, torque 450')")

if user_input:
    # Display user message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Extract numbers using regex
    battery_match = re.search(r"battery\s*(\d+\.?\d*)", user_input.lower())
    speed_match = re.search(r"speed\s*(\d+\.?\d*)", user_input.lower())
    torque_match = re.search(r"torque\s*(\d+\.?\d*)", user_input.lower())

    if battery_match and speed_match and torque_match:
        battery = float(battery_match.group(1))
        top_speed = float(speed_match.group(1))
        torque = float(torque_match.group(1))

        # Prepare data for prediction
        new_data = pd.DataFrame([{
            'top_speed_kmh': top_speed,
            'battery_capacity_kWh': battery,
            'torque_nm': torque
        }])

        pred = model.predict(new_data)[0]

        bot_response = f"🚗 Based on your input, the estimated EV range is *{pred:.2f} km*."
    else:
        bot_response = "⚠ Please provide values for battery, speed, and torque (e.g., 'battery 75, speed 200, torque 350')."

    # Display bot response
    with st.chat_message("assistant"):
        st.markdown(bot_response)

    st.session_state.messages.append({"role": "assistant", "content": bot_response})