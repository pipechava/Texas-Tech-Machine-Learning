# Optional 2b

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import time

start_time = time.time()

# Load the training and test data from the Pickle file
with open("vehicle_price_dataset_scaled.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

# Get some lengths
ncoeffs = train_data.shape[1]
nsamples = train_data.shape[0]

# Training constants
learning_rate = 0.25
n_iterations = 3000
print_step = 100

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

# Prediction
Y_pred = tf.add(tf.matmul(X, W), b)

# Predict new labels for test data
X_test = tf.constant(test_data.values.astype(np.float32))
Y_pred_test = tf.add(tf.matmul(X_test, W), b)

# Error
error = Y_pred - Y
mse = tf.reduce_mean(tf.square(error))

# Error for test
Y_true_test = tf.constant(test_labels.values.reshape(-1,1).astype(np.float32))
test_mse = tf.reduce_mean(tf.square(Y_pred_test - Y_true_test))

# Optimize MSE through gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
update_op = optimizer.minimize(mse)
#update_op_test = optimizer.minimize(test_mse)

# Create TensorFlow session and initialize it
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

iteration = 0
while iteration < n_iterations:
    # Run one iteration of the computation session to update coefficients
    _, mse_val = sess.run([update_op, mse])
    if (iteration % print_step == 0):
        print("iteration {:4d}:  MSE: {:.1f}".format(iteration, mse_val))
    iteration += 1

# Run one iteration of the computation session to update coefficients
mse_val_test = sess.run(test_mse)
print("Test MSE: {:.1f}".format(mse_val_test))


# Run a session to retrieve the coefficients & bias
W_result, b_result = sess.run([W, b])

# Print the coefficients
#print("Coeffs:", W_result)
#print("Bias:", b_result)

# Compute the training RMSE
training_rmse = mse_val ** .5
print("Training RMSE: {:.1f}".format(training_rmse))

# Compute the training RMSE Test
test_rmse = mse_val_test ** .5
print("Test RMSE: {:.1f}".format(test_rmse))

print("--- %s seconds ---" % (time.time() - start_time))