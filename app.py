import streamlit as st
import pandas as pd
import numpy as np
import os
from src.data_preprocessing import load_and_preprocess_data, prepare_input_data
from src.model_training import train_model, save_model, load_model
from src.model_evaluation import evaluate_model, predict_admission_chance

# Load the data
X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data('data/jamboree_admission.csv')

# Check if model exists, if not, train and save it
model_path = 'models/linear_regression_model.pkl'
if not os.path.exists(model_path):
    st.info("Model not found. Training a new model...")
    model = train_model(X_train, y_train, model_type='linear')
    save_model(model, model_path)
    st.success("New model trained and saved successfully!")
else:
    model = load_model(model_path)

st.title('Jamboree Admission Prediction')

# Sidebar for user input
st.sidebar.header('Input Parameters')
gre_score = st.sidebar.slider('GRE Score', 260, 340, 300)
toefl_score = st.sidebar.slider('TOEFL Score', 0, 120, 100)
university_rating = st.sidebar.slider('University Rating', 1, 5, 3)
sop = st.sidebar.slider('SOP', 1.0, 5.0, 3.0)
lor = st.sidebar.slider('LOR', 1.0, 5.0, 3.0)
cgpa = st.sidebar.slider('CGPA', 6.0, 10.0, 8.0)
research = st.sidebar.selectbox('Research', [0, 1])

# Make prediction
input_data = {
    'GRE Score': gre_score,
    'TOEFL Score': toefl_score,
    'University Rating': university_rating,
    'SOP': sop,
    'LOR ': lor,
    'CGPA': cgpa,
    'Research': research
}
input_df = prepare_input_data(input_data, scaler)
prediction = predict_admission_chance(model, input_df)

st.header('Admission Chance Prediction')
st.write(f'The predicted chance of admission is: {prediction:.2f}')

# Model performance
st.header('Model Performance')
performance = evaluate_model(model, X_test, y_test)
st.write(f"MSE: {performance['MSE']:.4f}")
st.write(f"RMSE: {performance['RMSE']:.4f}")
st.write(f"R2 Score: {performance['R2']:.4f}")

# Retrain model
st.header('Retrain Model')
model_type = st.selectbox('Select Model Type', ['linear', 'ridge', 'lasso'])
if model_type in ['ridge', 'lasso']:
    alpha = st.slider('Alpha', 0.01, 10.0, 1.0)
    params = {'alpha': [alpha]}
else:
    params = None

if st.button('Retrain Model'):
    new_model = train_model(X_train, y_train, model_type, params)
    save_model(new_model, model_path)
    st.success('Model retrained and saved successfully!')
    
    # Update performance metrics
    new_performance = evaluate_model(new_model, X_test, y_test)
    st.write('New Model Performance:')
    st.write(f"MSE: {new_performance['MSE']:.4f}")
    st.write(f"RMSE: {new_performance['RMSE']:.4f}")
    st.write(f"R2 Score: {new_performance['R2']:.4f}")
    
    # Update the current model
    model = new_model