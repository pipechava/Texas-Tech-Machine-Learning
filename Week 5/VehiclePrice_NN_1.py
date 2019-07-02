# Introduction to Artificial Intelligence
# Neural Network regression of vehicle price in TensorFlow, part 1
# By Juan Carlos Rojas

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

# Load the training and test data from the Pickle file
with open("vehicle_price_dataset_scaled.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

# Get some lengths
n_inputs = train_data.shape[1]
nsamples = train_data.shape[0]

# Training constants
batch_size = 60
learning_rate = 1e-5
n_epochs = 100
print_step = 5
n_nodes_l1 = 100

n_batches = int(np.ceil(nsamples / batch_size))

# Print the configuration
print("Batch size: {} Num batches: {} Num epochs: {} Learning rate: {}".format(batch_size, n_batches, n_epochs, learning_rate))
print("Num nodes in L1: {} Activation function: elu".format(n_nodes_l1))

# Input vector placeholders. Length is unspecified.
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
Y = tf.placeholder(tf.float32, shape=(None, 1), name="Y")

# Hidden layer 1:
#   Inputs: n_inputs
#   Outputs: n_nodes_l1
#   Activation: elu
W_L1 = tf.Variable(tf.truncated_normal([n_inputs, n_nodes_l1], stddev=1/np.sqrt(n_inputs)))
b_L1 = tf.Variable(tf.zeros(n_nodes_l1)) 
Y_L1 = tf.nn.elu(tf.add(tf.matmul(X, W_L1), b_L1))

# Output layer:
#   Inputs: n_nodes_l1
#   Outputs: 1
#   Activation: linear
W_L2 = tf.Variable(tf.truncated_normal([n_nodes_l1, 1], stddev=1/np.sqrt(n_nodes_l1)))
b_L2 = tf.Variable(tf.zeros(1)) 
Y_L2_linear = tf.add(tf.matmul(Y_L1, W_L2), b_L2)

# Cost function (MSE)
cost = tf.reduce_mean(tf.square(Y_L2_linear - Y))

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
        Y_batch = train_labels[dataidx:(dataidx+batch_size)].values.reshape(-1,1)
        feed_dict = {X: X_batch, Y: Y_batch}

        # Run one iteration of the computation session to update coefficients
        _, cost_vals[batch] = sess.run([update_op, cost], feed_dict=feed_dict)

        batch += 1
       

    # Print statistics about the cost measured in all the batches
    if (epoch % print_step == 0):
        print("Epoch {:4d}: Min cost: {:8.0f} Max cost: {:8.0f} Avg cost: {:8.0f}".format(epoch,\
            np.min(cost_vals),
            np.max(cost_vals),
            np.mean(cost_vals)))
    epoch += 1


# Run a session to compute the predictions against the training data
feed_dict = {X: train_data, Y: train_labels.values.reshape(-1,1)}
mse_training = sess.run(cost, feed_dict=feed_dict)

# Run a session to compute the predictions against the test data
feed_dict = {X: test_data, Y: test_labels.values.reshape(-1,1)}
mse_test = sess.run(cost, feed_dict=feed_dict)

print("Test RMSE:     {:.1f}".format(mse_test ** .5))
print("Training RMSE: {:.1f}".format(mse_training ** .5))
