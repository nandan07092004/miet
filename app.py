import streamlit as st
import pickle
import numpy as np


with open('logistic_model.pkl', 'rb') as f:
    log_model = pickle.load(f)
with open('knn_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)
with open('regression_model.pkl', 'rb') as f:
    reg_model = pickle.load(f)


with open('mappings.pkl', 'rb') as f:
    gender_map, course_map, year_map, cgpa_map, marital_map = pickle.load(f)



st.title("Mental Health & Productivity Analyzer")


gender_options = ['Male', 'Female']
course_options = ['Engineering', 'Information Technology', 'Arts', 'Islamic education', 'Law']
year_options = ['year 1', 'year 2', 'year 3', 'year 4']
cgpa_options = ['0 - 1.99', '2.00 - 2.49', '2.50 - 2.99', '3.00 - 3.49', '3.50 - 4.00']
marital_options = ['Yes', 'No']


gender = st.selectbox("Gender", gender_options)
age = st.number_input("Age", min_value=15, max_value=50)
course = st.selectbox("Course", course_options)
year = st.selectbox("Year of Study", year_options)
cgpa = st.selectbox("CGPA Range", cgpa_options)
marital = st.selectbox("Marital Status", marital_options)


depression_input = st.selectbox("Do you have Depression?", ['Yes', 'No'])
anxiety_input = st.selectbox("Do you have Anxiety?", ['Yes', 'No'])


gender_enc = gender_map.get(gender, -1)
course_enc = course_map.get(course, -1)
year_enc = year_map.get(year, -1)
cgpa_enc = cgpa_map.get(cgpa, -1)
marital_enc = marital_map.get(marital, -1)





features = [[gender_enc, age, course_enc, year_enc, cgpa_enc, marital_enc]]


if st.button("Analyze"):
    depression_pred = log_model.predict(features)[0]
    anxiety_pred = knn_model.predict(features)[0]
    productivity_pred = reg_model.predict(features)[0]

    st.subheader(" Analysis Result:")
    st.write("**Depression**:", "Yes" if depression_pred else "No")
    st.write("**Anxiety**:", "Yes" if anxiety_pred else "No")
    st.write("**Expected Productivity Score**:", round(productivity_pred, 2))
    

    st.write("---")
    st.write("**User reported:**")
    st.write(f"Depression: {depression_input}")
    st.write(f"Anxiety: {anxiety_input}")
