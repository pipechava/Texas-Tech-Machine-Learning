# Practice 3

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

print("Predictions:")
for idx in range(20):
    print("\tPredicted: {:.3f}\t Correct: {:6d}"\
          .format(Y_pred_proba[idx], test_labels.values[idx]))
    
# Binarize the predictions by comparing to a threshold
threshold = [0.35, 0.30, 0.25, 0.20, 0.15, 0.10]

for i in threshold:
    print("Threshold: ", i)
    Y_pred = (Y_pred_proba > i).astype(np.int_)
    
    # Compute the confusion matrix
    cmatrix = sklearn.metrics.confusion_matrix(test_labels, Y_pred)
    print("\tConfusion Matrix:")
    print(cmatrix)
    
    #compute the accuracy
    accuracy = sklearn.metrics.accuracy_score(test_labels, Y_pred)
    print("\tAccuracy: ", accuracy)
    
    #compute the precision
    precision = sklearn.metrics.precision_score(test_labels, Y_pred)
    print("\tPrecision: {:.3f}".format(precision))
    
    #compute the recall
    recall = sklearn.metrics.recall_score(test_labels, Y_pred)
    print("\tRecall: {:.3f}".format(recall))


### CORTARLA A 0.4!!!!!!!!!!!!!!
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

# Predict new labels for test data
auc_score = sklearn.metrics.roc_auc_score(test_labels, Y_pred_proba)
print("AUC score: {:.3f}".format(auc_score))

# Get the prediction probabilities
Y_pred_proba_training = model.predict_proba(train_data[cols])[::,1]

# Predict new labels for train data
auc_score_training = sklearn.metrics.roc_auc_score(train_labels, Y_pred_proba_training)
print("Training AUC score: {:.3f}".format(auc_score_training))