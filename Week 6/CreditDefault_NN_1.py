# Introduction to Artificial Intelligence
# Neural network classifier for credit default using TensorFlow, part 1
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
n_nodes_l1 = 100
batch_size = 32
learning_rate = .1
n_epochs = 500
eval_step = 5

n_batches = int(np.ceil(nsamples / batch_size))

# Print the configuration
print("Batch size: {} Num batches: {} Num epochs: {} Learning rate: {}".format(batch_size, n_batches, n_epochs, learning_rate))
print("Num nodes in L1: {} Activation function: ELU".format(n_nodes_l1))

# TensorFlow constants

# Input vector placeholders. Length is unspecified.
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
Y = tf.placeholder(tf.float32, shape=(None, 1), name="Y")

# Hidden layer 1:
#   Inputs: n_inputs
#   Outputs: n_nodes_l1
#   Activation: ELU
W_L1 = tf.Variable(tf.truncated_normal([n_inputs, n_nodes_l1], stddev=2/np.sqrt(n_inputs)))
b_L1 = tf.Variable(tf.zeros(n_nodes_l1)) 
Y_L1 = tf.nn.elu(tf.add(tf.matmul(X, W_L1), b_L1))

# Output layer:
#   Inputs: n_nodes_l1
#   Outputs: 1
#   Activation: logistic
W_L2 = tf.Variable(tf.truncated_normal([n_nodes_l1, 1], stddev=1/np.sqrt(n_nodes_l1)))
b_L2 = tf.Variable(tf.zeros(1)) 
Y_L2_linear = tf.add(tf.matmul(Y_L1, W_L2), b_L2)

# Cost function, plus the sigmoid part of the prediction
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits( 
        logits = Y_L2_linear, labels = Y))

# Optimize cost through gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
update_op = optimizer.minimize(cost)

# Prediction probability values
Y_pred_proba_calc = tf.nn.sigmoid(Y_L2_linear)

# Create TensorFlow session and initialize it
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Initialize lists to hold the history of metrics per epoch
trn_cost_hist = []
test_cost_hist = []
trn_auroc_hist = []
test_auroc_hist = []

epoch = 0
while epoch < n_epochs:
    batch = 0

    # Save a vector of cost values per batch
    cost_vals = np.zeros(n_batches)

    while batch < n_batches:

        # Select the data for the next batch
        dataidx = batch * batch_size
        X_batch = train_data[dataidx:(dataidx+batch_size)]
        Y_batch = train_labels[dataidx:(dataidx+batch_size)].values.reshape(-1,1)
        feed_dict = {X: X_batch, Y: Y_batch}

        # Run one iteration of the computation session to update coefficients
        _, cost_vals[batch] = sess.run([update_op, cost], feed_dict=feed_dict)
        batch += 1

    # Evaluate and print the results so far
    if (epoch % eval_step == 0):

        # Compute the average cost for all mini-batches in this epoch
        trn_cost_avg = np.mean(cost_vals)

        # Compute the ROC AUC against the full training data
        feed_dict = {X: train_data, Y: train_labels.values.reshape(-1,1)}
        Y_pred_proba_train = sess.run(Y_pred_proba_calc, feed_dict=feed_dict)
        train_auroc = sklearn.metrics.roc_auc_score(train_labels, Y_pred_proba_train)

        # Compute the cost and ROC AUC against the test data
        feed_dict = {X: test_data, Y: test_labels.values.reshape(-1,1)}
        Y_pred_proba_test = sess.run(Y_pred_proba_calc, feed_dict=feed_dict)
        test_cost = sess.run(cost, feed_dict=feed_dict)
        test_auroc = sklearn.metrics.roc_auc_score(test_labels, Y_pred_proba_test)

        print("Epoch: {:4d} trn_cost: {:.5f} test_cost: {:.5f} trn_auroc: {:.4f} test_auroc: {:.4f}".\
              format(epoch, trn_cost_avg, test_cost, train_auroc, test_auroc))

        # Save the metrics to the history
        trn_cost_hist.append(trn_cost_avg)
        test_cost_hist.append(test_cost)
        trn_auroc_hist.append(train_auroc)
        test_auroc_hist.append(test_auroc)

    epoch += 1

# Print the best results (as if we had done early stopping)
epoch_hist = [i for i in range(0, n_epochs, eval_step)]

best_idx = test_auroc_hist.index(max(test_auroc_hist))
print("Max test ROC AUC:  {:.4f} at epoch: {}".format(test_auroc_hist[best_idx], epoch_hist[best_idx]))

best_idx = trn_auroc_hist.index(max(trn_auroc_hist))
print("Max train ROC AUC: {:.4f} at epoch: {}".format(trn_auroc_hist[best_idx], epoch_hist[best_idx]))

best_idx = test_cost_hist.index(min(test_cost_hist))
print("Min test cost:  {:.5f} at epoch: {}".format(test_cost_hist[best_idx], epoch_hist[best_idx]))

best_idx = trn_cost_hist.index(min(trn_cost_hist))
print("Min train cost: {:.5f} at epoch: {}".format(trn_cost_hist[best_idx], epoch_hist[best_idx]))

# Plot the metrics history
plt.plot(epoch_hist, trn_cost_hist, "b")
plt.plot(epoch_hist, test_cost_hist, "r")
plt.xlabel("epoch")
plt.ylabel("cost")
plt.title("Cost vs. epoch")
plt.figure()
plt.plot(epoch_hist, trn_auroc_hist, "b")
plt.plot(epoch_hist, test_auroc_hist, "r")
plt.xlabel("epoch")
plt.ylabel("ROC AUC")
plt.title("ROC AUC vs. epoch")
plt.show()



