# Introduction to Artificial Intelligence
# Logistic regression classifier for credit default dataset, part 1
# By Juan Carlos Rojas

import numpy as np
import pandas as pd
import pickle
import sklearn.linear_model
import sklearn.metrics
import matplotlib.pyplot as plt

# Load the training and test data from the Pickle file
with open("credit_card_default_dataset.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

# Select columns of interest
cols = train_data.columns

# Create and train a new logistic regression classifier
model = sklearn.linear_model.LogisticRegression(\
        solver='newton-cg', \
        tol=1e-4, max_iter=1000)

# Train it with the training data and labels
model.fit(train_data[cols], train_labels)

# Print some results
print("Iterations used: ", model.n_iter_)
print("Intercept: ", model.intercept_)
print("Coeffs: ", model.coef_)

# Get the prediction probabilities
Y_pred_proba = model.predict_proba(test_data[cols])[::,1]

# Binarize the predictions by comparing to a threshold
threshold = 0.3
print("Threshold: ", threshold)
Y_pred = (Y_pred_proba > threshold).astype(np.int_)

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

