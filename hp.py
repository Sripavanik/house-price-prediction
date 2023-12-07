import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
# Load your dataset
df = pd.read_csv('train (1).csv')


# Display the head of the dataset in Streamlit


# Select feature (X) and target variable (y)
x = df[['LotArea']].values
y = df['SalePrice']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(x_train, y_train)

st.title("House Price Prediction App")

# Add a slider for user input
lot_area = st.slider('LotArea', float(df['LotArea'].min()), float(df['LotArea'].max()), float(df['LotArea'].mean()))

# Convert the slider value to a numpy array and reshape for prediction
lot_area_arr = np.array(lot_area).reshape(1, -1)

# Make a prediction based on user input
prediction = model.predict(lot_area_arr)

st.subheader('Predicted House Price:')
st.write(prediction[0])

# Visualization
#st.subheader("Visualization")
#plt.scatter(y_test, model.predict(x_test))
#plt.xlabel("True prices")
#plt.ylabel("Predicted prices")
#plt.title("True prices vs Predicted prices")
#st.pyplot()
