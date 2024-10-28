import streamlit as st
import pickle
import numpy as np
import pandas as pd  # Make sure to import pandas

# Load the model and DataFrame
with open('pipe.pkl', 'rb') as f:
    pipe = pickle.load(f)

with open('df.pkl', 'rb') as f1:
    df = pickle.load(f1)

st.title("Laptop Price Predictor")

# Brand
company = st.selectbox('Brand', df['Company'].unique())

# Type of laptop
type_ = st.selectbox('Type', df['TypeName'].unique())

# RAM
ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# Weight
weight = st.number_input('Weight of the Laptop', min_value=0.0)

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS
ips = st.selectbox('IPS', ['No', 'Yes'])

# Screen size
screen_size = st.slider('Screen size in inches', 10.0, 18.0, 13.0)

# Resolution
resolution = st.selectbox('Screen Resolution', [
    '1920x1080', '1366x768', '1600x900', '3840x2160',
    '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'
])

# CPU
cpu = st.selectbox('CPU', df['Cpu Brand'].unique())

# HDD and SSD
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])

# GPU
gpu = st.selectbox('GPU', df['Gpu brand'].unique())

# OS
os = st.selectbox('OS', df['os'].unique())

# Prediction button
if st.button('Predict Price'):
    # Query
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

    # Prepare the query for prediction as a DataFrame
    query = pd.DataFrame([[company, type_, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os]], 
                        columns=['Company', 'TypeName', 'Ram(GB)', 'Weight(kg)', 'Touch Screen', 'Ips', 
                                    'ppi', 'Cpu Brand', 'HDD', 'SSD', 'Gpu brand', 'os'])

    try:
        predicted_price = int(np.exp(pipe.predict(query)[0]))
        st.title(f"The predicted price of this configuration is ${predicted_price}")
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
