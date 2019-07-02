# Introduction to Artificial Intelligence
# Linear regression of vehicle price in TensorFlow, part 1
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

# TensorFlow constants

# Input vectors
X = tf.constant(train_data.values.astype(np.float32))
Y = tf.constant(train_labels.values.reshape(-1,1).astype(np.float32))

# Solve the Normal equations: W = (X' * X)^-1 * X' * Y
XT = tf.transpose(X)
W = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), Y)

# Create TensorFlow session and initialize it
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Run the session
W_result = sess.run(W)
sess.close()

# Print the coefficients
print(W_result)
