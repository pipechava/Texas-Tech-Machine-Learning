# Practice 5

import numpy as np
import pandas as pd
import pickle
import sklearn.ensemble
import sklearn.metrics
import matplotlib.pyplot as plt

# Load the training and test data from the Pickle file
with open("credit_card_default_dataset.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

# Select columns of interest
cols = train_data.columns

i = 10

# cycle through various n_estimators
while i <= 200:
    print("For n_estimators = ", i)
    
    
    # Create and train a Random Forest classifier
    model = sklearn.ensemble.RandomForestClassifier(n_estimators=i,min_samples_leaf=350)

    # Train it with the training data and labels
    model.fit(train_data[cols], train_labels)
    
    # Get prediction probabilities
    Y_pred_proba = model.predict_proba(test_data[cols])[::,1]
    
    # Compute a precision & recall graph
    #precisions, recalls, thresholds = sklearn.metrics.precision_recall_curve(test_labels, Y_pred_proba)
    #plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    #plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    #plt.legend(loc="center left")
    #plt.xlabel("Threshold")
    #plt.show()
    
    # Plot a ROC curve (Receiver Operating Characteristic)
    # Compares true positive rate with false positive rate
    #fpr, tpr, _ = sklearn.metrics.roc_curve(test_labels, Y_pred_proba)
    #plt.plot(fpr,tpr)
    #plt.xlabel('False Positive Rate')
    #plt.ylabel('True Positive Rate')
    #plt.title('Receiver operating characteristic')
    #plt.show()
    
    #calculate area under the curve for test data
    auc_score = sklearn.metrics.roc_auc_score(test_labels, Y_pred_proba)
    print("\tTest AUC score: {:.3f}".format(auc_score))
    
    # Predict new labels for training data
    Y_pred_proba_training = model.predict_proba(train_data[cols])[::,1]
    
    #calculate area under the curve for train data
    auc_score_training = sklearn.metrics.roc_auc_score(train_labels, Y_pred_proba_training)
    print("\tTraining AUC score: {:.3f}".format(auc_score_training))
    
    i = i + 10
    
i = 5

# cycle through various leaf sizes
while i <= 35:
    print("Leaf size = ", i)
    
    # Create and train a Decision Tree classifier
    model = sklearn.ensemble.RandomForestClassifier(n_estimators=100,min_samples_leaf=i)

    # Train it with the training data and labels
    model.fit(train_data[cols], train_labels)
    
    # Get prediction probabilities
    Y_pred_proba = model.predict_proba(test_data[cols])[::,1]
    
    # Compute a precision & recall graph
    #precisions, recalls, thresholds = sklearn.metrics.precision_recall_curve(test_labels, Y_pred_proba)
    #plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    #plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    #plt.legend(loc="center left")
    #plt.xlabel("Threshold")
    #plt.show()
    
    # Plot a ROC curve (Receiver Operating Characteristic)
    # Compares true positive rate with false positive rate
    #fpr, tpr, _ = sklearn.metrics.roc_curve(test_labels, Y_pred_proba)
    #plt.plot(fpr,tpr)
    #plt.xlabel('False Positive Rate')
    #plt.ylabel('True Positive Rate')
    #plt.title('Receiver operating characteristic')
    #plt.show()
    
    # calculate the area under the curve for test data
    auc_score = sklearn.metrics.roc_auc_score(test_labels, Y_pred_proba)
    print("\tTest AUC score: {:.3f}".format(auc_score))
    
    # Predict new labels for training data
    Y_pred_proba_training = model.predict_proba(train_data[cols])[::,1]
    
    # calculate the area under the curve for train data
    auc_score_training = sklearn.metrics.roc_auc_score(train_labels, Y_pred_proba_training)
    print("\tTraining AUC score: {:.3f}".format(auc_score_training))
    
    i = i + 5

# Create and train a Decision Tree classifier with optimal n_estimator and leaf size
model = sklearn.ensemble.RandomForestClassifier(n_estimators=100,min_samples_leaf=20)

# Train it with the training data and labels
model.fit(train_data[cols], train_labels)
    
# Get prediction probabilities
Y_pred_proba = model.predict_proba(test_data[cols])[::,1]
    
# Compute a precision & recall graph
precisions, recalls, thresholds = sklearn.metrics.precision_recall_curve(test_labels, Y_pred_proba)
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

# calculate the area under the curve for test data
auc_score = sklearn.metrics.roc_auc_score(test_labels, Y_pred_proba)
print("\tTest AUC score: {:.3f}".format(auc_score))
    
# Predict new labels for training data
Y_pred_proba_training = model.predict_proba(train_data[cols])[::,1]

# calculate the area under the curve for train data
auc_score_training = sklearn.metrics.roc_auc_score(train_labels, Y_pred_proba_training)
print("\tTraining AUC score: {:.3f}".format(auc_score_training))
