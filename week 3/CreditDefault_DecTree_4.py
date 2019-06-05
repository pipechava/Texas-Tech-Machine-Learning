# Introduction to Artificial Intelligence
# Decision Tree classifier for credit default dataset, part 4
# By Juan Carlos Rojas

import numpy as np
import pandas as pd
import pickle
import sklearn.tree
import sklearn.metrics
import matplotlib.pyplot as plt

# Load the training and test data from the Pickle file
with open("credit_card_default_dataset.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

# Select columns of interest
cols = train_data.columns

# Create and train a Decision Tree classifier
model = sklearn.tree.DecisionTreeClassifier(min_samples_leaf=350)

# Train it with the training data and labels
model.fit(train_data[cols], train_labels)

# Get prediction probabilities
Y_pred_proba = model.predict_proba(test_data[cols])[::,1]

auc_score = sklearn.metrics.roc_auc_score(test_labels, Y_pred_proba)
print("AUC score: {:.3f}".format(auc_score))

# Predict new labels for training data
Y_pred_proba_training = model.predict_proba(train_data[cols])[::,1]
auc_score_training = sklearn.metrics.roc_auc_score(\
    train_labels, Y_pred_proba_training)
print("Training AUC score: {:.3f}".format(auc_score_training))

# Export the decision tree
sklearn.tree.export_graphviz(model, out_file="dtree_4.dot", feature_names=cols) 
