# Introduction to Artificial Intelligence
# Linear regression classifier for credit default dataset, part 4
# By Juan Carlos Rojas

import numpy as np
import pandas as pd
import pickle
import sklearn.linear_model

# Load the training and test data from the Pickle file
with open("credit_card_default_dataset.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

# Select columns of interest
cols = train_data.columns

# Create and train a new LinearRegression model
model = sklearn.linear_model.LinearRegression()

# Train it with the training data and labels
model.fit(train_data[cols], train_labels)

# Create a list of the abs(coeff) by feature
coeff_abs_list = []
for idx in range(len(model.coef_)):
    coeff_abs_list.append( (abs(model.coef_[idx]), cols[idx]) )

# Sort the list
coeff_abs_list.sort(reverse=True)

# Print the coefficients in order
for idx in range(len(model.coef_)):
    print("Feature: {:26s} abs(coef): {:.4f}".format(coeff_abs_list[idx][1], coeff_abs_list[idx][0]))

