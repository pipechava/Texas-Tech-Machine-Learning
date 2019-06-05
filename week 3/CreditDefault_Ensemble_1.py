# Introduction to Artificial Intelligence
# Ensemble classifier, part 1
# By Juan Carlos Rojas

import numpy as np
import pandas as pd
import pickle
import sklearn.linear_model
import sklearn.neighbors
import sklearn.ensemble
import sklearn.metrics
import matplotlib.pyplot as plt

# Load the training and test data from the Pickle file
with open("credit_card_default_dataset.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

# Select columns of interest
cols = train_data.columns

# Create a logistic regression classifier
logistic = sklearn.linear_model.LogisticRegression(\
        solver='newton-cg', \
        tol=1e-4, max_iter=1000)

# Create a k-Nearest neighbors classifier
kNN = sklearn.neighbors.KNeighborsClassifier(n_neighbors=70)

# Create a Random Forest classifier
randforest = sklearn.ensemble.RandomForestClassifier(\
    n_estimators=100,
    min_samples_leaf=20)

# Create a voting ensemble of classifiers
model = sklearn.ensemble.VotingClassifier(
    estimators=[('logistic', logistic),
                ('kNN', kNN),
                ('randforest', randforest),
                ],voting='soft')

# Train it with the training data and labels
model.fit(train_data[cols], train_labels)

# Get prediction probabilities
Y_pred_proba = model.predict_proba(test_data[cols])[::,1]

"""
# Compute a precision & recall graph
precisions, recalls, thresholds = \
    sklearn.metrics.precision_recall_curve(test_labels, Y_pred_proba)
plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
plt.legend(loc="center left")
plt.xlabel("Threshold")
plt.show()

# Plot a ROC curve (Receiver Operating Characteristic)
# Compares true positive rate with false positive rate
fpr, tpr, _ = sklearn.metrics.roc_curve(test_labels, Y_pred_proba)
plt.plot(fpr,tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.show()
"""

auc_score = sklearn.metrics.roc_auc_score(test_labels, Y_pred_proba)
print("AUC score: {:.3f}".format(auc_score))

# Predict new labels for training data
#"""
Y_pred_proba_training = model.predict_proba(train_data[cols])[::,1]
auc_score_training = sklearn.metrics.roc_auc_score(\
    train_labels, Y_pred_proba_training)
print("Training AUC score: {:.3f}".format(auc_score_training))
#"""
