import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model with caching
@st.cache_resource
def load_model():
    return joblib.load("printer_predictive_model.pkl")

model = load_model()

# Set page layout and title
st.set_page_config(page_title="3D Printer Predictive Maintenance", layout="centered")
st.title("🧠 Predictive Maintenance for 3D Printers")
st.markdown("Enter sensor data from your 3D printer to predict potential faults.")

# Sidebar info
with st.sidebar:
    st.header("ℹ️ About")
    st.write("This model predicts specific faults in 3D printers using 9 sensor readings.")
    st.markdown("[📂 GitHub Repository](https://github.com/vedant1741/predictive-maintenance)")

# Input form
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        index = st.number_input("ID / Record Number (`Unnamed: 0`)", value=0, step=1)
        x_temp = st.number_input("X Stepper Temp (°C)", value=40.0)
        y_temp = st.number_input("Y Stepper Temp (°C)", value=40.0)
        z_temp = st.number_input("Z Stepper Temp (°C)", value=40.0)
        extruder_temp = st.number_input("Extruder Temp (°C)", value=200.0)
        hotend_temp = st.number_input("Hotend Temp (°C)", value=210.0)

    with col2:
        bed_temp = st.number_input("Bed Temp (°C)", value=60.0)
        current = st.number_input("MainBoard Current (A)", value=1.5)
        vibration = st.number_input("Vibration Level", value=0.02, step=0.001)
        speed = st.number_input("Print Speed (mm/s)", value=50.0)

    submitted = st.form_submit_button("🚀 Predict Fault")

if submitted:
    input_data = pd.DataFrame([[
        x_temp, y_temp, z_temp,
        extruder_temp, hotend_temp,
        bed_temp, current, vibration, speed
    ]], columns=[
        'X_Stepper_Temp', 'Y_Stepper_Temp', 'Z_Stepper_Temp',
        'Extruder_Temp', 'Hotend_Temp',
        'Bed_Temp', 'Current_MainBoard',
        'Vibration_Level', 'Print_Speed'
    ])

    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)

    st.markdown("### 🧾 Prediction Result")
    st.success(f"📌 Predicted Fault: **{prediction}**")

    with st.expander("🔍 View Input Data"):
        st.dataframe(input_data)

    with st.expander("📊 View Probabilities"):
        proba_df = pd.DataFrame(prediction_proba, columns=model.classes_)
        st.dataframe(proba_df.style.highlight_max(axis=1, color='lightgreen'))
