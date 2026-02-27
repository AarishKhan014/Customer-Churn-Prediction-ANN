import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import pickle

# Loading Model
model = tf.keras.models.load_model('model.h5')

# Loading Scaler and Encoders
with open('onehot_encoder_gender.pkl', 'rb') as file:
    onehot_gender = pickle.load(file)

with open('onehot_encoder_geography.pkl', 'rb') as file:
    onehot_geography = pickle.load(file)

with open('standard_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


# Streamlit App

st.title ('Customer Churn Prediction')
st.write('by:- AARISH KHAN')

credit_score = st.number_input('Credit Score')
geography = st.selectbox('Geography', onehot_geography.categories_[0])
gender = st.selectbox('Gender', onehot_gender.categories_[0])
age = st.slider('Age', 18, 100)
tenure = st.slider('Tenure', 0, 10)
balance = st.number_input('Balance')
num_of_products = st.slider('Number Of Products', 1, 4)
has_credit_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
estimated_salary = st.number_input('Estimated_Salary')




# Preparing Data

input_data = pd.DataFrame([{
    'CreditScore' : credit_score,
    'Geography' : geography,
    'Gender' : gender,
    'Age' : age,
    'Tenure' : tenure,
    'Balance' : balance,
    'NumOfProducts' : num_of_products,
    'HasCrCard' : has_credit_card,
    'IsActiveMember' : is_active_member,
    'EstimatedSalary' : estimated_salary
}])


geo_encoder = onehot_geography.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoder, columns=onehot_geography.get_feature_names_out())

gen_encoder = onehot_gender.transform([[gender]]).toarray()
gen_encoded_df = pd.DataFrame(gen_encoder, columns=onehot_gender.get_feature_names_out())


input_data = pd.concat([input_data, geo_encoded_df, gen_encoded_df], axis=1).drop(columns=['Geography', 'Gender'])

input_scaled = scaler.transform(input_data)

prediction = model.predict(input_scaled)

pred_proba = prediction[0][0]

st.write(pred_proba)

if pred_proba > 0.5:
    st.write('The Customer Is Likely To To Churn')
else:
    st.write('The Customer Is Not Likely To Churn')