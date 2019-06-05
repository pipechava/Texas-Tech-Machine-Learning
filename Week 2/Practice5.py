#Practice 5

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#load pickle data
with open("vehicle_price_dataset.pickle", "rb") as f:
    train_data, train_labels, test_data, test_labels = pickle.load(f)

#print("Train labels values: \n{}".format(train_labels))
#print("Test labels values: \n{}".format(test_labels))

# Insert a column of ones
train_data["ones"] = 1

# Select all columns 
X = train_data.values
Y = train_labels.values

# Solve the Normal equations: W = (X' * X)^-1 * X' * Y
XT = np.transpose(X)
W = np.linalg.inv(XT @ X) @ XT @ Y

intercept = W[0]
coeffs = W[1:]

# Get the headers of data
cols = train_data.columns

print("intercept: {:.3f}".format(intercept))

# adds each feature (header) with its  value to a list of tuples
features = []
for i in range(len(coeffs)):
    features.append((cols[i], coeffs[i]))

# print the value of each coefficient for each feature (each of the headers)
print("coeffs:")
for i in features:
    print("\t{:28s}: {:.3f}".format(i[0], i[1]))

# Predict new labels for test data
test_data["ones"] = 1
Xn = test_data.values
Y_pred = Xn @ W

# Predict new labels for train data
Y_pred2 = X @ W
#print("print y pred:", Y_pred2)
#print("print Y:", Y)

# Print the first few predictions
print("predictions for test data")
for idx in range(10):
    print("\tPredicted: {:6.0f} Correct: {:6d}"\
          .format(Y_pred[idx], test_labels.values[idx]))

print("predictions for train data")
for idx2 in range(10):
    print("\tPredicted: {:6.0f} Correct: {:6d}" \
          .format(Y_pred2[idx2], train_labels.values[idx2]))

# Compute the root mean squared error
error = Y_pred - test_labels.values
error2 = Y_pred2 - train_labels.values
rmse = (error ** 2).mean() ** .5
rmse2 = (error2 ** 2).mean() ** .5
print("Training RMSE: {:.2f}".format(rmse2))
print("Test RMSE: {:.2f}".format(rmse))