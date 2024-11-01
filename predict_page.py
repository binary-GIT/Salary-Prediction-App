import streamlit as st
import pickle
import numpy as np
import pandas as pd

@st.cache_data
def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

def show_predict_page():
    st.title("Software Developer Salary Prediction")
    st.write("### We need some information to predict the salary")

    countries = (
        "United States", "India", "United Kingdom", "Germany",
        "Canada", "Brazil", "France", "Spain", "Australia",
        "Netherlands", "Poland", "Italy", "Russian Federation", "Sweden",
    )

    education_levels = (
        "Less than a Bachelors", "Bachelor’s degree",
        "Master’s degree", "Post grad",
    )

    country = st.selectbox("Country", countries)
    education = st.selectbox("Education Level", education_levels)

    experience = st.slider("Years of Experience", 0, 50, 3)

    ok = st.button("Calculate Salary")
    if ok:
        # Transform country
        country_transformed = le_country.transform([country])[0] if country in le_country.classes_ else le_country.transform([le_country.classes_[0]])[0]
        # Transform education
        education_transformed = le_education.transform([education])[0] if education in le_education.classes_ else le_education.transform([le_education.classes_[0]])[0]

        # Prepare the input DataFrame with the correct feature names
        X_df = pd.DataFrame({
            'Country': [country_transformed],
            'EdLevel': [education_transformed],
            'YearsCodePro': [experience]
        })

        # Predict the salary using the DataFrame
        salary = regressor.predict(X_df)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")


