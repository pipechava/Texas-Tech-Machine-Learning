# Introduction to Artificial Intelligence
# Neural Network classification of the MNIST database in TensorFlow, part 1
# By Juan Carlos Rojas

import pickle
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn.metrics
import tensorflow as tf

# Load the training and test data from the Pickle file
with open("mnist_dataset_unscaled.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

# Scale the training and test data
pixel_mean = np.mean(train_data)
pixel_std = np.std(train_data)
train_data = (train_data - pixel_mean) / pixel_std
test_data = (test_data - pixel_mean) / pixel_std

# One-hot encode the labels
encoder = sklearn.preprocessing.OneHotEncoder(sparse=False, categories='auto')
train_labels_onehot = encoder.fit_transform(train_labels.reshape(-1, 1))
test_labels_onehot = encoder.transform(test_labels.reshape(-1, 1))
num_classes = len(encoder.categories_[0])

# Get some lengths
n_inputs = train_data.shape[1]
nsamples = train_data.shape[0]

# Training constants
n_nodes_l1 = 100
batch_size = 32
learning_rate = .03
n_epochs = 50
eval_step = 5

n_batches = int(np.ceil(nsamples / batch_size))

# Print the configuration
print("Batch size: {} Num batches: {} Num epochs: {} Learning rate: {}".format(batch_size, n_batches, n_epochs, learning_rate))
print("Num nodes in L1: {} Activation function: tanh".format(n_nodes_l1))

# Input vector placeholders. Length is unspecified.
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
Y = tf.placeholder(tf.float32, shape=(None, num_classes), name="Y")

# Hidden layer 1:
#   Inputs: n_inputs
#   Outputs: n_nodes_l1
#   Activation: tanh
W_L1 = tf.Variable(tf.truncated_normal([n_inputs, n_nodes_l1], stddev=1/np.sqrt(n_inputs)))
b_L1 = tf.Variable(tf.zeros(n_nodes_l1)) 
Y_L1 = tf.nn.tanh(tf.add(tf.matmul(X, W_L1), b_L1))

# Output layer:
#   Inputs: n_nodes_l1
#   Outputs: num_classes
#   Activation: softmax
W_L2 = tf.Variable(tf.truncated_normal([n_nodes_l1, num_classes], stddev=1/np.sqrt(n_nodes_l1)))
b_L2 = tf.Variable(tf.zeros(num_classes)) 
Y_L2_linear = tf.add(tf.matmul(Y_L1, W_L2), b_L2)

# Prediction values 
Y_pred_proba = tf.nn.softmax(Y_L2_linear)
Y_pred_calc = tf.argmax(Y_pred_proba, 1)

# Cost function, plus the softmax part of the prediction
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2( 
                    logits = Y_L2_linear, labels = Y))

# Optimize cost through gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
update_op = optimizer.minimize(cost)

# Create TensorFlow session and initialize it
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

epoch = 0
while epoch < n_epochs:
    batch = 0

    # Save a vector of cost values per batch
    cost_vals = np.zeros(n_batches)
    
    while batch < n_batches:

        # Select the data for the next batch
        dataidx = batch * batch_size
        X_batch = train_data[dataidx:(dataidx+batch_size)]
        Y_batch = train_labels_onehot[dataidx:(dataidx+batch_size)]
        feed_dict = {X: X_batch, Y: Y_batch}

        # Run one iteration of the computation session to update coefficients
        _, cost_vals[batch] = sess.run([update_op, cost], feed_dict=feed_dict)
        batch += 1

    # Evaluate and print the results so far
    if (epoch % eval_step == 0):

        # Compute the average cost for all mini-batches in this epoch
        trn_cost_avg = np.mean(cost_vals)
        print("Epoch: {:4d} trn_cost: {:.5f}".format(epoch, trn_cost_avg))
    epoch += 1

# Run a session to compute the predictions against the full training data
feed_dict = {X: train_data, Y: train_labels_onehot}
Y_pred_training = sess.run(Y_pred_calc, feed_dict=feed_dict)

# Run a session to compute the predictions against the test data
feed_dict = {X: test_data, Y: test_labels_onehot}
Y_pred_test = sess.run(Y_pred_calc, feed_dict=feed_dict)

# Accuracy, precision & recall
print("Accuracy:   {:.4f}".format(sklearn.metrics.accuracy_score(test_labels, Y_pred_test)))
print("Precision:  {:.4f}".format(sklearn.metrics.precision_score(test_labels, Y_pred_test, average='weighted')))
print("Recall:     {:.4f}".format(sklearn.metrics.recall_score(test_labels, Y_pred_test, average='weighted')))

# Compute the prediction accuracy against the training data
print("Against training set:")
print("  Accuracy:   {:.4f}".format(sklearn.metrics.accuracy_score(train_labels, Y_pred_training)))
print("  Precision:  {:.4f}".format(sklearn.metrics.precision_score(train_labels, Y_pred_training, average='weighted')))
print("  Recall:     {:.4f}".format(sklearn.metrics.recall_score(train_labels, Y_pred_training, average='weighted')))

