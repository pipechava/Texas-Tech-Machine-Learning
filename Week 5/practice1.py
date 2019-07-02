# Practice 1

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

# Load the training and test data from the Pickle file
with open("vehicle_price_dataset_scaled.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

# Insert a column of ones
train_data["ones"] = 1›
test_data["ones"] = 1

# TensorFlow constants

# Input vectors
X = tf.constant(train_data.values.astype(np.float32))
Y = tf.constant(train_labels.values.reshape(-1,1).astype(np.float32))

# Solve the Normal equations: W = (X' * X)^-1 * X' * Y
XT = tf.transpose(X)
W = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), Y)

# Predict new labels for test data
Xn = tf.constant(test_data.values.astype(np.float32))
Y_pred = tf.matmul(Xn, W)

# Create TensorFlow session and initialize it
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Run the session
W_result = sess.run(W)
Y_pred_result = sess.run(Y_pred)

#store Y_pred_result in a pandas data frame named Y_pred_result_df
Y_pred_result_df = pd.DataFrame.from_records(Y_pred_result)
sess.close()

# Print the first few predictions
for idx in range(10):
    # use [0].iloc[idx] for first column and idx row
    print("Predicted: {:6.0f} Correct: {:6d}".format(Y_pred_result_df[0].iloc[idx], test_labels.values[idx]))