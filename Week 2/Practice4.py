#Practice 4

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

# Select columns odometer and ones
cols = ["ones", "odometer"]
X = train_data[cols].values
Y = train_labels.values

# Solve the Normal equations: W = (X' * X)^-1 * X' * Y
XT = np.transpose(X)
W = np.linalg.inv(XT @ X) @ XT @ Y
#print(W)

eq_str = "price = {:.3f} + {:.3f} * odometer".format(W[0], W[1])
print(eq_str)

# Predict new labels for test data
test_data["ones"] = 1
Xn = test_data[cols].values
Y_pred = Xn @ W

# Print the first few predictions
for idx in range(10):
    print("Predicted: {:6.0f} Correct: {:6d}"\
          .format(Y_pred[idx], test_labels.values[idx]))

# Compute the root mean squared error
error = Y_pred - test_labels.values
rmse = (error ** 2).mean() ** .5
print("RMSE: {:.2f}".format(rmse))