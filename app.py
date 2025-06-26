import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

# Set page configuration for a wider layout and custom title
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('one_hot_encoder_geo.pkl', 'rb') as file:
    one_hot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5em;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5em;
    }
    .section-header {
        font-size: 1.5em;
        color: #424242;
        margin-top: 1em;
    }
    .prediction-box {
        padding: 1em;
        border-radius: 10px;
        text-align: center;
        margin-top: 1em;
    }
    .churn-high {
        background-color: #FFCDD2;
        color: #D32F2F;
    }
    .churn-low {
        background-color: #C8E6C9;
        color: #2E7D32;
    }
    .sidebar .stSelectbox, .sidebar .stSlider, .sidebar .stNumberInput {
        margin-bottom: 1em;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit app
st.markdown('<div class="main-title">Customer Churn Prediction</div>', unsafe_allow_html=True)
st.markdown("Enter the customer details below to predict the likelihood of churn.")

# Sidebar for user inputs
with st.sidebar:
    st.header("Customer Details")
    
    # Organize inputs in a clean layout
    geography = st.selectbox('Geography', one_hot_encoder_geo.categories_[0], help="Select the customer's geographical location")
    gender = st.selectbox('Gender', label_encoder_gender.classes_, help="Select the customer's gender")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider('Age', 18, 92, 30, help="Select the customer's age")
        tenure = st.slider('Tenure (Years)', 0, 10, 5, help="Years the customer has been with the bank")
        num_of_products = st.slider('Number of Products', 1, 4, 1, help="Number of bank products used")
    with col2:
        credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=600, help="Customer's credit score")
        balance = st.number_input('Balance ($)', min_value=0.0, value=0.0, step=1000.0, help="Customer's account balance")
        estimated_salary = st.number_input('Estimated Salary ($)', min_value=0.0, value=50000.0, step=1000.0, help="Customer's estimated annual salary")
    
    has_cr_card = st.selectbox('Has Credit Card', [1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No', help="Does the customer have a credit card?")
    is_active_member = st.selectbox('Is Active Member', [1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No', help="Is the customer an active member?")

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode Geography
geo_encoded = one_hot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict churn
if st.button("Predict Churn", type="primary"):
    prediction = model.predict(input_data_scaled)
    prediction_prob = prediction[0][0]
    
    # Display results in a styled container
    with st.container():
        st.markdown('<div class="section-header">Prediction Result</div>', unsafe_allow_html=True)
        st.markdown(f"**Churn Probability:** {prediction_prob:.2%}")
        
        # Conditional styling based on prediction
        if prediction_prob > 0.5:
            st.markdown(
                '<div class="prediction-box churn-high">⚠️ The customer is likely to churn.</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="prediction-box churn-low">✅ The customer is not likely to churn.</div>',
                unsafe_allow_html=True
            )