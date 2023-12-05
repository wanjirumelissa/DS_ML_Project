#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import os

# Print current working directory and list of files
print("Current Working Directory:", os.getcwd())
print("Files in Current Directory:", os.listdir())

# Load the model
model_path = "best_model2.pkl"
loaded_model = joblib.load(model_path)

# Streamlit App
st.title("Linear Regression Model Explorer")

# Upload CSV data through Streamlit
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Display the first few rows of the DataFrame
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Feature selection and scaling
    features = ['employ', 'car', 'carcat', 'age', 'inccat']
    df_selected = df[features]
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_selected), columns=features)

    # Prediction using the linear regression model
    st.subheader("Linear Regression Model Prediction")

    # Make predictions
    predictions = loaded_model.predict(df_scaled)
    df["Predicted Income"] = predictions
    st.write(df[["income", "Predicted Income"]])

    # Scatter plot of predicted vs. actual income
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='income', y='Predicted Income', data=df)
    plt.title('Predicted vs. Actual Income')
    plt.xlabel('Actual Income')
    plt.ylabel('Predicted Income')
    st.pyplot()

    # Model coefficients visualization
    st.subheader("Model Coefficients")
    coef_df = pd.DataFrame({'Feature': features, 'Coefficient': loaded_model.coef_})
    st.bar_chart(coef_df.set_index('Feature'))

    # Evaluation metrics
    st.subheader("Model Evaluation Metrics")
    y_test = df["income"]
    y_pred = loaded_model.predict(df_scaled)
    st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    st.write("R-squared:", r2_score(y_test, y_pred))


# In[ ]:




