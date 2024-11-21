import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Title and description
st.title("3MTT Impact for Good Hackathon")
st.title("SmartCrop")
st.write("Input soil metrics to predict the best crop.")

# User inputs
nitrogen = st.number_input("Nitrogen content ratio", min_value=0.0, max_value=100.0, step=0.1)
phosphorous = st.number_input("Phosphorous content ratio", min_value=0.0, max_value=100.0, step=0.1)
potassium = st.number_input("Potassium content ratio", min_value=0.0, max_value=100.0, step=0.1)
ph_value = st.number_input("pH value", min_value=0.0, max_value=14.0, step=0.1)

# Load dataset and train model (typically you'd do this outside the user input section)
data = pd.read_csv("soil_measures.csv")
X = data.drop(columns="crop")
y = data["crop"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict button
if st.button("Predict"):
    input_data = [[nitrogen, phosphorous, potassium, ph_value]]
    prediction = model.predict(input_data)
    st.write(f"Recommended Crop: {prediction[0]}")