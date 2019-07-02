# Introduction to Artificial Intelligence
# Neural network classifier for credit default using TensorFlow, part 4
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
n_inputs = train_data.shape[1]
nsamples = train_data.shape[0]

# Training constants
batch_size = 60
learning_rate = .02
n_epochs = 1000
eval_step = 50
n_nodes_l1 = 100
regularization_scale = 5e-4

# TensorFlow constants

# Input vectors
X = tf.constant(train_data.values.astype(np.float32))
Y = tf.constant(train_labels.values.reshape(-1,1).astype(np.float32))

n_batches = int(np.ceil(nsamples / batch_size))

# Print the configuration
print("Batch size: {} Num batches: {} Num epochs: {} Learning rate: {}".format(batch_size, n_batches, n_epochs, learning_rate))
print("Num nodes in L1: {} Activation function: ReLU".format(n_nodes_l1))
print("Regularization scale: {}  Regularization type: L2".format(regularization_scale))

# TensorFlow constants

# Input vector placeholders. Length is unspecified.
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
Y = tf.placeholder(tf.float32, shape=(None, 1), name="Y")

# Hidden layer 1:
#   Inputs: n_inputs
#   Outputs: n_nodes_l1
#   Activation: ReLU
W_L1 = tf.Variable(tf.truncated_normal([n_inputs, n_nodes_l1], stddev=1/np.sqrt(n_inputs)))
b_L1 = tf.Variable(tf.zeros(n_nodes_l1)) 
Y_L1 = tf.nn.relu(tf.add(tf.matmul(X, W_L1), b_L1))

# Output layer:
#   Inputs: n_nodes_l1
#   Outputs: 1
#   Activation: logistic
W_L2 = tf.Variable(tf.truncated_normal([n_nodes_l1, 1], stddev=1/np.sqrt(n_nodes_l1)))
b_L2 = tf.Variable(tf.zeros(1)) 
Y_L2_linear = tf.add(tf.matmul(Y_L1, W_L2), b_L2)

# Cost function, plus the sigmoid part of the prediction
base_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits( 
                    logits = Y_L2_linear, labels = Y))
regularization_cost = tf.reduce_sum(tf.square(W_L1)) + tf.reduce_sum(tf.square(W_L2))
cost = regularization_cost * regularization_scale + base_cost

# Optimize cost through gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
update_op = optimizer.minimize(cost)

# Prediction probability values
Y_pred_proba_calc = tf.nn.sigmoid(Y_L2_linear)

# Create TensorFlow session and initialize it
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

epoch = 0
while epoch < n_epochs:
    batch = 0

    while batch < n_batches:

        # Select the data for the next batch
        dataidx = batch * batch_size
        X_batch = train_data[dataidx:(dataidx+batch_size)]
        Y_batch = train_labels[dataidx:(dataidx+batch_size)].values.reshape(-1,1)
        feed_dict = {X: X_batch, Y: Y_batch}

        # Run one iteration of the computation session to update coefficients
        _ = sess.run(update_op, feed_dict=feed_dict)
        batch += 1

    # Evaluate the test score every certain number of epochs
    if (epoch % eval_step == 0):

        # Compute the test accuracy
        feed_dict = {X: test_data, Y: test_labels.values.reshape(-1,1)}
        Y_pred_proba_test = sess.run(Y_pred_proba_calc, feed_dict=feed_dict)
        auc_score = sklearn.metrics.roc_auc_score(test_labels, Y_pred_proba_test)
        print("Epoch {:4d}: Test AUC score: {:.4f}".format(epoch, auc_score))
        
    epoch += 1

# Run a session to compute the predictions against the training data
feed_dict = {X: train_data, Y: train_labels.values.reshape(-1,1)}
Y_pred_proba_training = sess.run(Y_pred_proba_calc, feed_dict=feed_dict)

# Run a session to compute the predictions against the test data
feed_dict = {X: test_data, Y: test_labels.values.reshape(-1,1)}
Y_pred_proba_test = sess.run(Y_pred_proba_calc, feed_dict=feed_dict)

auc_score = sklearn.metrics.roc_auc_score(test_labels, Y_pred_proba_test)
print("Test AUC score: {:.3f}".format(auc_score))

# Predict new labels for training data
auc_score_training = sklearn.metrics.roc_auc_score(\
    train_labels, Y_pred_proba_training)
print("Training AUC score: {:.3f}".format(auc_score_training))
