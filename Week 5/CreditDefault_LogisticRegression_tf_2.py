# Introduction to Artificial Intelligence
# Logistic regression classifier for credit default using TensorFlow, part 2
# By Juan Carlos Rojas

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import sklearn.metrics
import matplotlib.pyplot as plt

# Load the training and test data from the Pickle file
with open("credit_card_default_dataset.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

# Get some lengths
ncoeffs = train_data.shape[1]
nsamples = train_data.shape[0]

# Training constants
learning_rate = .5
n_iterations = 10000
print_step = 1000

# TensorFlow constants

# Input vectors
X = tf.constant(train_data.values.astype(np.float32))
Y = tf.constant(train_labels.values.reshape(-1,1).astype(np.float32))

# Initial coefficients & bias

# Start with uniformly random values
W = tf.Variable(tf.random_uniform([ncoeffs, 1], -1.0, 1.0))

# Start the bias with zeros
b = tf.Variable(0.0)

# TensorFlow computations

# Linear portion of the prediction model (without the sigmoid)
Y_linear = tf.add(tf.matmul(X, W), b)

# Cost function
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits( 
                    logits = Y_linear, labels = Y))

# Optimize cost through gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
update_op = optimizer.minimize(cost)

# Prediction against training set (with the sigmoid)
Y_pred_training = tf.nn.sigmoid(Y_linear)

# Prediction against the test test
X_test = tf.constant(test_data.values.astype(np.float32))
Y_pred_test = tf.nn.sigmoid(tf.add(tf.matmul(X_test, W), b))

# Create TensorFlow session and initialize it
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

iteration = 0
while iteration < n_iterations:
    # Run one iteration of the computation session to update coefficients
    _, cost_val = sess.run([update_op, cost])
    if (iteration % print_step == 0):
        print("iteration {:5d}:  Cost: {}".format(iteration, cost_val))
    iteration += 1

# Run a session to retrieve the last set of predictions against the training data
Y_pred_proba_training = sess.run(Y_pred_training)

# Run a session to retrieve the last set of predictions against the test data
Y_pred_proba = sess.run(Y_pred_test)

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
print("Test AUC score: {:.3f}".format(auc_score))

# Predict new labels for training data
auc_score_training = sklearn.metrics.roc_auc_score(\
    train_labels, Y_pred_proba_training)
print("Training AUC score: {:.3f}".format(auc_score_training))
