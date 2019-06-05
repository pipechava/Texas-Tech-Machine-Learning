# Introduction to Artificial Intelligence
# Multivariate linear regression model, part 4
# By Juan Carlos Rojas

import numpy as np
import pandas as pd
import pickle
import sklearn.linear_model
import sklearn.metrics

# Load the training and test data from the Pickle file
with open("vehicle_price_dataset_scaled.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

# Select columns of interest
cols = train_data.columns

# Create and train a new LinearRegression model
model = sklearn.linear_model.LinearRegression()

# Train it with the training data and labels
model.fit(train_data[cols], train_labels)

# Print the coefficients
print(model.intercept_)
print(model.coef_)

# Predict new labels for test data
Y_pred = model.predict(test_data[cols])

# Compute the root mean squared error 
mse = sklearn.metrics.mean_squared_error(test_labels.values, Y_pred)
rmse = mse ** .5
print("RMSE: {:.2f}".format(rmse))

