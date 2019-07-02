# Introduction to Artificial Intelligence
# Linear regression of vehicle price in TensorFlow, part 3
# By Juan Carlos Rojas

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

# Load the training and test data from the Pickle file
with open("vehicle_price_dataset_scaled.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

# Insert a column of ones
train_data["ones"] = 1

# Get some lengths
ncoeffs = train_data.shape[1]
nsamples = train_data.shape[0]

# Training constants
learning_rate = .1
n_iterations = 10000
print_step = 1000

# TensorFlow constants

# Input vectors
X = tf.constant(train_data.values.astype(np.float32))
Y = tf.constant(train_labels.values.reshape(-1,1).astype(np.float32))

# Initial coefficients
# Start with uniformly random values
W = tf.Variable(tf.random_uniform([ncoeffs, 1], -1.0, 1.0))

# TensorFlow computations

# Prediction
Y_pred = tf.matmul(X, W)

# Error
error = Y_pred - Y
mse = tf.reduce_mean(tf.square(error))

# Optimize MSE through gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
update_op = optimizer.minimize(mse)

# Create TensorFlow session and initialize it
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

iteration = 0
while iteration < n_iterations:
    # Run one iteration of the computation session to update coefficients
    _, mse_val = sess.run([update_op, mse])
    if (iteration % print_step == 0):
        print("iteration {:4d}:  MSE: {}".format(iteration, mse_val))
    iteration += 1

# Run a session to retrieve the coefficients
W_result = sess.run(W)

# Print the coefficients
print(W_result)

# Compute the Training RMSE
training_rmse = mse_val ** .5
print("Training RMSE: ", training_rmse)

