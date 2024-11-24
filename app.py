# Model deployment on steamlit web app


import pickle
from tensorflow.keras.models import load_model
import pandas as pd
import random
import numpy as np
import streamlit as st



# Load the model 
model = load_model("model.keras")


# load Label encoder and scaler

with open("scaler.pkl" , "rb") as f:
    scaler = pickle.load(file=f)

with open("encoders.pkl" , "rb") as f:
    encoders = pickle.load(file=f)


# Initialize the Streamlit app
st.title("Customter Churn prediction")

geography = st.selectbox("Geography" , encoders['Geography'].classes_)
gender = st.selectbox("Gender" , encoders['Gender'].classes_)
age = st.slider('Age' , 0 , 100)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated_salary")
tenure = st.slider("Tenure" , 0 , 10)
num_of_products = st.slider("Number of Products" , 0 ,14)
has_cr_card = st.selectbox("Has Credit Card" , [0,1])
is_active_member = st.selectbox("Is Active Member" , [0,1])


input_data = pd.DataFrame({
    'CreditScore' : [credit_score],
    'Geography' :[encoders['Geography'].transform([geography])],
    'Gender' : [encoders['Gender'].transform([[gender]])],
    'Age':[age],
    'Tenure' : [tenure],
    'Balance' :[balance],
    'NumOfProducts' :[num_of_products],
    "HasCrCard" : [has_cr_card],
    "IsActivemember" :[is_active_member],
    "EstimatedSalary" :[estimated_salary]
})



input_data_scaled = scaler.fit_transform(input_data)


prediction = model.predict(input_data_scaled)

prediction_probability = np.where(prediction > 0.5 , 1 , 0)

st.write(f"Churn probability :-{prediction_probability}")


if prediction_probability > 0.5 :
    st.write("The customer is likely to churn !!!")
else:
    st.write("This customer is not likely to churn !!!")