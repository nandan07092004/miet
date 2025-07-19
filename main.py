import streamlit as st
import pickle
import numpy as np

# Load trained models
with open('logistic_model.pkl', 'rb') as f:
    log_model = pickle.load(f)
with open('knn_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)
with open('regression_model.pkl', 'rb') as f:
    reg_model = pickle.load(f)

# Load mappings
with open('mappings.pkl', 'rb') as f:
    gender_map, course_map, year_map, cgpa_map, marital_map = pickle.load(f)

# Title
st.title("MindMate: Mental Health & Productivity Profiler")

# Input fields
gender_options = ['Male', 'Female']
course_options = ['Engineering', 'IT', 'Science', 'Commerce', 'Islamic education', 'Others', 'BIT']
year_options = ['year 1', 'year 2', 'year 3', 'year 4', 'Masters']
marital_options = ['Yes', 'No']

gender = st.selectbox("Select your gender", gender_options)
age = st.number_input("Enter your age", min_value=15, max_value=50)
course = st.selectbox("Select your course", course_options)
year = st.selectbox("Select your current year of study", year_options)

# CGPA input as float from 0.0 to 10.0
cgpa_raw = st.slider("Enter your CGPA (0.0 - 10.0)", min_value=0.0, max_value=10.0, step=0.01)

marital = st.selectbox("Are you married?", marital_options)

# Map CGPA to corresponding range from training
if cgpa_raw <= 2.49:
    cgpa_enc = 0
elif 2.50 <= cgpa_raw <= 2.99:
    cgpa_enc = 1
elif 3.00 <= cgpa_raw <= 3.49:
    cgpa_enc = 2
else:
    cgpa_enc = 3

# Encode other categorical features
gender_enc = gender_map.get(gender, -1)
course_enc = course_map.get(course, -1)
year_enc = year_map.get(year, -1)
marital_enc = marital_map.get(marital, -1)

# Final input feature array
features = [[gender_enc, age, course_enc, year_enc, cgpa_enc, marital_enc]]

# Prediction
if st.button("Analyze"):
    depression_pred = log_model.predict(features)[0]
    anxiety_pred = knn_model.predict(features)[0]
    productivity_pred = reg_model.predict(features)[0]

    st.subheader("Prediction Result")
    st.write("**Depression**:", "Yes" if depression_pred else "No")
    st.write("**Anxiety**:", "Yes" if anxiety_pred else "No")
    st.write("**Expected Productivity Score**:", round(productivity_pred, 2))
