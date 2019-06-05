# Introduction to Artificial Intelligence
# Linear regression classifier for credit default dataset, part 2
# By Juan Carlos Rojas

import numpy as np
import pandas as pd
import pickle
import sklearn.linear_model
import sklearn.metrics

# Load the training and test data from the Pickle file
with open("credit_card_default_dataset.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

# Select columns of interest
cols = train_data.columns

# Create and train a new LinearRegression model
model = sklearn.linear_model.LinearRegression()

# Train it with the training data and labels
model.fit(train_data[cols], train_labels)

# Print the coefficients
#print(model.intercept_)
#print(model.coef_)

# Predict new labels for test data
Y_pred_proba = model.predict(test_data[cols])

# Binarize the predictions by comparing to a threshold
threshold = 0.5
print("Threshold: ", threshold)
Y_pred = (Y_pred_proba > threshold).astype(np.int_)

# Count how many are predicted as 0 and 1
print("Predicted as 1: ", np.count_nonzero(Y_pred))
print("Predicted as 0: ", len(Y_pred) - np.count_nonzero(Y_pred))

# Compute the statistics
cmatrix = sklearn.metrics.confusion_matrix(test_labels, Y_pred)
print("Confusion Matrix:")
print(cmatrix)

accuracy = sklearn.metrics.accuracy_score(test_labels, Y_pred)
print("Accuracy: {:.3f}".format(accuracy))

precision = sklearn.metrics.precision_score(test_labels, Y_pred)
print("Precision: {:.3f}".format(precision))

recall = sklearn.metrics.recall_score(test_labels, Y_pred)
print("Recall: {:.3f}".format(recall))

