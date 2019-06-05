# Introduction to Artificial Intelligence
# Linear regression model of 1 variable, part 7
# By Juan Carlos Rojas

import numpy as np
import pandas as pd
import pickle

# Load the training and test data from the Pickle file
with open("vehicle_price_dataset.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

# Insert a column of ones
train_data["ones"] = 1
test_data["ones"] = 1

# Create a list of tuples of RMSE values for each feature
results_list = []

# Loop through all columns in the train data
for sel_col in train_data.columns:
    if sel_col == "ones":
        continue
    
    # Select columns of interest
    cols = ["ones", sel_col]
    X = train_data[cols].values
    Y = train_labels.values

    # Solve the Normal equations: W = (X' * X)^-1 * X' * Y
    XT = np.transpose(X)
    W = np.linalg.inv(XT @ X) @ XT @ Y

    # Predict new labels for test data
    Xn = test_data[cols].values
    Y_pred = Xn @ W

    # Compute the root mean squared error 
    error = Y_pred - test_labels.values
    rmse = (error ** 2).mean() ** .5
    
    #print("Feature: {:26s} RMSE: {:.2f}".format(sel_col, rmse))
    results_list.append( (rmse, sel_col) )

# Sort the list
results_list.sort()
# Print the first 15 
for idx in range(15):
    print("Feature: {:26s} RMSE: {:.2f}".format(results_list[idx][1], results_list[idx][0]))

