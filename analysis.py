#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
file_path = 'C:/Users/admin/Documents/Income_Project_ADS/dataset1.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the DataFrame
print(df.head())

# Display general information about the dataset
print(df.info())

# Display summary statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Use the pd.get_dummies() function for one-hot encoding
df_encoded = pd.get_dummies(df, columns=['marital', 'ed', 'jobsat', 'gender'], drop_first=True)

# Display the first few rows of the DataFrame after encoding
print(df_encoded.head())

# Select relevant features
features = ['employ', 'car', 'carcat', 'age', 'inccat']

# Split the data into features (X) and target variable (y)
X = df[features]
y = df['income']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
linear_regression_model = LinearRegression()

# Fit the model to the training data
linear_regression_model.fit(X_train, y_train)

# Display the coefficients and intercept
print("Coefficients:", linear_regression_model.coef_)
print("Intercept:", linear_regression_model.intercept_)

# Make predictions on the test set
y_pred = linear_regression_model.predict(X_test)

# Evaluation metrics
from sklearn.metrics import mean_squared_error, r2_score

print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))

# Create a summary of the OLS regression results
X_train_with_intercept = sm.add_constant(X_train)
ols_model = sm.OLS(y_train, X_train_with_intercept).fit()
print(ols_model.summary())

# Save the best model using joblib
best_model = linear_regression_model
joblib.dump(best_model, "C:/Users/admin/Documents/Income_Project_ADS/best_model2.pkl")


# In[ ]:




