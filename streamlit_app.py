import streamlit as st
import pandas as pd
import joblib

st.title("3D Printer Predictive Maintenance - AI Model")

@st.cache_resource
def load_model():
    return joblib.load('printer_predictive_model.pkl')

model = load_model()

st.sidebar.header("Upload your CSV data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file for prediction", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data", data)
    if st.button("Predict"):
        predictions = model.predict(data)
        st.write("Predictions", predictions)
        st.write(pd.DataFrame({'Prediction': predictions}))
else:
    st.write("Please upload a CSV file with the same features as the training data (excluding the target column).") 