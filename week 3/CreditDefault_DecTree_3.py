# Introduction to Artificial Intelligence
# Decision Tree classifier for credit default dataset, part 3
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
model = sklearn.tree.DecisionTreeClassifier(min_samples_leaf=.016)

# Train it with the training data and labels

#print("Trained with :24000")
#model.fit(train_data[cols][:24000], train_labels[:24000])

#print("Trained with :16000")
#model.fit(train_data[cols][:16000], train_labels[:16000])

#print("Trained with :12000")
#model.fit(train_data[cols][:12000], train_labels[:12000])

print("Trained with 12000:")
model.fit(train_data[cols][12000:], train_labels[12000:])

# Get prediction probabilities
Y_pred_proba = model.predict_proba(test_data[cols])[::,1]

auc_score = sklearn.metrics.roc_auc_score(test_labels, Y_pred_proba)
print("Test AUC score: {:.3f}".format(auc_score))

