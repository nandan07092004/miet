import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier


df = pd.read_csv('mental_health_data.csv')


df.dropna(inplace=True)


gender_map = {'Male': 0, 'Female': 1}
course_map = {
    'Engineering': 0, 'IT': 1, 'Science': 2, 'Commerce': 3,
    'Islamic education': 4, 'Others': 5, 'BIT': 6
}
year_map = {
    'year 1': 0, 'year 2': 1, 'year 3': 2, 'year 4': 3, 'Masters': 4
}
cgpa_map = {
    '0.00 - 2.49': 0, '2.50 - 2.99': 1, '3.00 - 3.49': 2, '3.50 - 4.00': 3
}
marital_map = {'Yes': 1, 'No': 0}
depression_map = {'Yes': 1, 'No': 0}
anxiety_map = {'Yes': 1, 'No': 0}




df['gender_enc'] = df['Choose your gender'].replace(gender_map)
df['course_enc'] = df['What is your course?'].replace(course_map)
df['year_enc'] = df['Your current year of Study'].replace(year_map)
df['cgpa_enc'] = df['What is your CGPA?'].replace(cgpa_map)
df['marital_enc'] = df['Marital status'].replace(marital_map)


df['course_enc'] = pd.to_numeric(df['course_enc'], errors='coerce').fillna(-1).astype(int)
df['year_enc'] = pd.to_numeric(df['year_enc'], errors='coerce').fillna(-1).astype(int)
df['cgpa_enc'] = pd.to_numeric(df['cgpa_enc'], errors='coerce').fillna(-1).astype(int)
df['gender_enc'] = pd.to_numeric(df['gender_enc'], errors='coerce').fillna(-1).astype(int)
df['marital_enc'] = pd.to_numeric(df['marital_enc'], errors='coerce').fillna(-1).astype(int)


y_depression = df['Do you have Depression?'].replace(depression_map)
y_anxiety = df['Do you have Anxiety?'].replace(anxiety_map)
y_depression = pd.to_numeric(y_depression, errors='coerce').fillna(-1).astype(int)
y_anxiety = pd.to_numeric(y_anxiety, errors='coerce').fillna(-1).astype(int)
y_productivity = df['cgpa_enc'] * 20 


X = df[['gender_enc', 'Age', 'course_enc', 'year_enc', 'cgpa_enc', 'marital_enc']]


lr = LogisticRegression()
knn = KNeighborsClassifier(n_neighbors=3)
reg = LinearRegression()


lr.fit(X, y_depression)
knn.fit(X, y_anxiety)
reg.fit(X, y_productivity)


with open('logistic_model.pkl', 'wb') as f:
    pickle.dump(lr, f)
with open('knn_model.pkl', 'wb') as f:
    pickle.dump(knn, f)
with open('regression_model.pkl', 'wb') as f:
    pickle.dump(reg, f)


with open('mappings.pkl', 'wb') as f:
    pickle.dump((gender_map, course_map, year_map, cgpa_map, marital_map), f)

print("Models and mappings saved successfully.")