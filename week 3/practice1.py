# Practice 1

import numpy as np
import pandas as pd
import pickle
import sklearn.linear_model
import sklearn.metrics
import matplotlib.pyplot as plt

# Load the training and test data from the Pickle file
with open("credit_card_default_dataset.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

# Get the headers of data
cols = train_data.columns

# Create and train a new LinearRegression model
model = sklearn.linear_model.LinearRegression()

# Train it with the training data and labels
model.fit(train_data[cols], train_labels)

# Print the coefficients
print("intercept: {:.3f}".format(model.intercept_))

# adds each feature (header) with its  value to a list of tuples
features2 = []
for i in range(len(model.coef_)):
    features2.append((cols[i], model.coef_[i]))

# print the value of each coefficient for each feature (each of the headers)
print("coeffs:")
for i in features2:
    print("\t{:28s}: {:.3f}".format(i[0], i[1]))

# Predict new labels for test data
Y_pred_proba = model.predict(test_data[cols])

print("Predictions:")
for idx in range(20):
    print("\tPredicted: {:.3f}\t Correct: {:6d}"\
          .format(Y_pred_proba[idx], test_labels.values[idx]))
