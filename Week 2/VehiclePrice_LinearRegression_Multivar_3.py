# Introduction to Artificial Intelligence
# Multivariate linear regression model, part 3
# By Juan Carlos Rojas

import numpy as np
import pandas as pd
import pickle

# Load the training and test data from the Pickle file
# Use the scaled dataset
with open("vehicle_price_dataset_scaled.pickle", "rb") as f:
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

# Create a list of the abs(coeff) by feature
coeff_abs_list = []
for idx in range(len(W)):
    coeff_abs_list.append( (abs(W[idx]), cols[idx]) )
# Sort the list
coeff_abs_list.sort(reverse=True)

# Try different number of coefficients
for n_coeffs_to_use in range(5,71,5):

    # Pick the first n features from the sorted list, as the columns of interest
    cols = [coeff_abs_list[idx][1] for idx in range(n_coeffs_to_use)]
    X = train_data[cols].values
    Y = train_labels.values

    # Solve the Normal equations: W = (X' * X)^-1 * X' * Y
    XT = np.transpose(X)
    W = np.linalg.inv(XT @ X) @ XT @ Y

    # Predict new labels for test data
    test_data["ones"] = 1
    Xn = test_data[cols].values
    Y_pred = Xn @ W

    # Compute the root mean squared error 
    error = Y_pred - test_labels.values
    rmse = (error ** 2).mean() ** .5
    print("Using {:2d} coefficients, RMSE: {:.2f}".format(n_coeffs_to_use, rmse))

