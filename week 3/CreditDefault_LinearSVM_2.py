# Introduction to Artificial Intelligence
# Linear SVM classifier for credit default dataset, part 2
# By Juan Carlos Rojas

import numpy as np
import pandas as pd
import pickle
import sklearn.svm
import sklearn.calibration
import sklearn.metrics
import matplotlib.pyplot as plt

# Load the training and test data from the Pickle file
with open("credit_card_default_dataset.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

# Select columns of interest
cols = train_data.columns

# Create and train a linear SVM model
linear_svm_model = sklearn.svm.LinearSVC(\
        C=1,\
        loss="hinge", \
        class_weight="balanced",\
        tol=1e-3, max_iter=100000, verbose=0)

# Wrap the linear SVM classifier inside a generic classifier that
# lets us convert decision functions into prediction probabilities
wrapper_classifier = sklearn.calibration.CalibratedClassifierCV(linear_svm_model, cv=2) 

# Train it with the training data and labels
wrapper_classifier.fit(train_data[cols], train_labels)

# Get the prediction probabilities
Y_pred_proba = wrapper_classifier.predict_proba(test_data[cols])[::,1]

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

auc_score = sklearn.metrics.roc_auc_score(test_labels, Y_pred_proba)
print("AUC score: {:.3f}".format(auc_score))

# Predict new labels for training data
Y_pred_proba_training = wrapper_classifier.predict_proba(train_data[cols])[::,1]
auc_score_training = sklearn.metrics.roc_auc_score(\
    train_labels, Y_pred_proba_training)
print("Training AUC score: {:.3f}".format(auc_score_training))
