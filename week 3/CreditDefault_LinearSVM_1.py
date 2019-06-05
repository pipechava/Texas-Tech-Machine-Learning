# Introduction to Artificial Intelligence
# Linear SVM classifier for credit default dataset, part 1
# By Juan Carlos Rojas

import numpy as np
import pandas as pd
import pickle
import sklearn.svm
import sklearn.metrics

# Load the training and test data from the Pickle file
with open("credit_card_default_dataset.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

# Select columns of interest
cols = train_data.columns

# Create and train a linear SVM model
model = sklearn.svm.LinearSVC(\
        C=1, loss="hinge", class_weight="balanced",\
        tol=1e-3, max_iter=100000)

# Train it with the training data and labels
model.fit(train_data[cols], train_labels)

# Print some results
print("Iterations used: ", model.n_iter_)
print("Intercept: ", model.intercept_)
print("Coeffs: ", model.coef_)

# Get the predictions
Y_pred = model.predict(test_data[cols])

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

