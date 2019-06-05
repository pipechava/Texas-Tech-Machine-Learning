# Introduction to Artificial Intelligence
# Multivariate linear regression model 
# By Juan Carlos Rojas

import numpy as np
import pandas as pd
import pickle

# Load the training and test data from the Pickle file
with open("vehicle_price_dataset.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

# Insert a column of ones
train_data["ones"] = 1

# Select columns of interest
cols = train_data.columns
X = train_data[cols].values
Y = train_labels.values

# Solve the Normal equations: W = (X' * X)^-1 * X' * Y
XT = np.transpose(X)
W = np.linalg.inv(XT @ X) @ XT @ Y
print(W)

# Predict new labels for test data
test_data["ones"] = 1
Xn = test_data[cols].values
Y_pred = Xn @ W

# Compute the root mean squared error 
error = Y_pred - test_labels.values
rmse = (error ** 2).mean() ** .5
print("RMSE: {:.2f}".format(rmse))
