from turtle import color
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

data = pd.read_csv("c:/Users/archa/Downloads/diabetes - diabetes.csv")
df= pd.read_csv("c:/Users/archa/Downloads/diabetes - diabetes.csv")
z_scores = np.abs((data - data.mean()) / data.std())


z_threshold = 3
outliers_z = (z_scores > z_threshold)


Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1


outliers_iqr = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR)))


outliers_combined = outliers_z | outliers_iqr


outliers_indices = outliers_combined.any(axis=1)


cleaned_data = data[~outliers_indices]


original_shape = data.shape
cleaned_shape = cleaned_data.shape

print("Original shape:", original_shape)
print("Cleaned shape:", cleaned_shape)
st.set_page_config(page_title="GlucoAlert", page_icon="https://cdn-icons-png.flaticon.com/512/6326/6326243.png", layout="centered")
st.markdown(
        f"""
        <style>
         .you{{color:white; background-color:rgb(158, 9, 36)}}
         .my{{color:white;background-color:green}}
        .stApp {{
            background-image: url("https://images.unsplash.com/photo-1576091160550-2173dba999ef?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
            background-size: cover;
            opacity: 0.5;
        }}
           .dashboard-image {{
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 400px;
            height: 300px;
             max-width:650px;
          
        }}    
        
        </style>
        """,
        unsafe_allow_html=True
    )
st.title('Diabetes Checkup')
st.markdown('<img src="https://static.vecteezy.com/system/resources/thumbnails/010/794/157/small_2x/3d-rendering-medical-report-with-blood-tube-pills-and-stethoscope-signing-health-checklist-medical-check-up-report-3d-render-illustration-png.png" class="dashboard-image">', unsafe_allow_html=True)

st.sidebar.title('Patient Data')
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

def user_report():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    blood_pressure = st.sidebar.slider('Blood Pressure', 0, 122, 70)
    skin_thickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
    insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    bmi = st.sidebar.slider('BMI', 0, 67, 20)
    diabetes_pedigree_function = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.47)
    age = st.sidebar.slider('Age', 21, 88, 33)

    user_report_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree_function,
        'Age': age
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data
user_data = user_report()
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200,random_state=42))
model.fit(x_train, y_train)
accuracy_score(y_test,model.predict(x_test))

st.subheader('Accuracy:')
st.write(str(accuracy_score(y_test, model.predict(x_test))))


user_result = model.predict(user_data)
st.subheader("Your Report")
output = ''
if user_result[0] == 0:
    st.markdown('<h2 class="my">Congratulations ! You are not diabetic.</h2>',unsafe_allow_html=True)
else:
    st.markdown('<h2 class="you">Danger! You are diabetic.</h2>',unsafe_allow_html=True)
st.write(output)


