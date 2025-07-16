import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model once using caching
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# App title and description
st.set_page_config(page_title="Predictive Maintenance", layout="centered")

st.title("🔧 Predictive Maintenance for Machines")
st.markdown("This AI-based tool predicts whether a machine is likely to fail based on sensor data.")

# Sidebar info
with st.sidebar:
    st.header("ℹ️ About")
    st.write("This app uses a Random Forest model trained on historical sensor data to predict machine failures.")
    st.write("Developed by Vedant Joshi")
    st.markdown("[📂 View Source Code](https://github.com/vedant1741/predictive-maintenance)")

st.subheader("📊 Input Machine Sensor Readings")

# Input form
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        air_temp = st.number_input("🌡️ Air Temperature (K)", value=300.0)
        process_temp = st.number_input("🔥 Process Temperature (K)", value=310.0)
        torque = st.number_input("🔩 Torque (Nm)", value=40.0)
    
    with col2:
        rot_speed = st.number_input("⚙️ Rotational Speed (rpm)", value=1500)
        tool_wear = st.number_input("🛠️ Tool Wear (min)", value=100)
        type_map = {"L": 0, "M": 1, "H": 2}
        type_input = st.selectbox("🧪 Product Type", options=list(type_map.keys()))

    submitted = st.form_submit_button("🚀 Predict Failure")

if submitted:
    input_data = np.array([[air_temp, process_temp, rot_speed, torque, tool_wear, type_map[type_input]]])
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.markdown("### 🧾 Prediction Result")
    if prediction == 1:
        st.error("⚠️ Machine is likely to **FAIL**. Please schedule maintenance.")
    else:
        st.success("✅ Machine is **OK**. No immediate maintenance needed.")

    st.markdown(f"**Model Confidence:** {prob*100:.2f}%")

    with st.expander("🔍 View Input Data"):
        st.write(pd.DataFrame(input_data, columns=["Air Temp", "Process Temp", "Rot Speed", "Torque", "Tool Wear", "Type"]))
