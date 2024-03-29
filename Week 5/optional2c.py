# Optional 2c

import pickle
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn.metrics
import tensorflow as tf
import time

start_time = time.time()

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
learning_rate = 100
n_iterations = 5000
print_step = 100

# TensorFlow constants

# Input data matrices
X = tf.constant(train_data.astype(np.float32))
Y = tf.constant(train_labels_onehot.astype(np.float32))

# Initial coefficients & bias

# Start with random values that follow a normal distribution (Xavier's initialization)
W = tf.Variable(tf.truncated_normal([n_inputs, num_classes], stddev=1/np.sqrt(n_inputs)))

# Start with zero biases
b = tf.Variable(tf.zeros(num_classes)) 

# TensorFlow computations

# Linear portion of the prediction model (without the softmax)
Y_linear = tf.add(tf.matmul(X, W), b)

# Cost function, plus the softmax part of the prediction
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2( 
                    logits = Y_linear, labels = Y))

# Optimize cost through gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
update_op = optimizer.minimize(cost)

# Prediction against training set (with the softmax)
Y_pred_proba_training = tf.nn.softmax(Y_linear)
Y_pred_training_calc = tf.argmax(Y_pred_proba_training, 1)

# Prediction against the test test
X_test = tf.constant(test_data.astype(np.float32))
Y_pred_proba_test = tf.nn.softmax(tf.add(tf.matmul(X_test, W), b))
Y_pred_test_calc = tf.argmax(Y_pred_proba_test, 1)

# Create TensorFlow session and initialize it
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

iteration = 0
while iteration < n_iterations:
    # Run one iteration of the computation session to update coefficients
    _, cost_val = sess.run([update_op, cost])
    if (iteration % print_step == 0):
        print("iteration {:4d}:  Cost: {:.4f}".format(iteration, cost_val))
    iteration += 1

# Run a session to compute the predictions against the training data
Y_pred_training = sess.run(Y_pred_training_calc)

# Run a session to compute the predictions against the test data
Y_pred_test = sess.run(Y_pred_test_calc)

# Accuracy, precision & recall
print("Accuracy:   {:.4f}".format(sklearn.metrics.accuracy_score(test_labels, Y_pred_test)))
print("Precision:  {:.4f}".format(sklearn.metrics.precision_score(test_labels, Y_pred_test, average='weighted')))
print("Recall:     {:.4f}".format(sklearn.metrics.recall_score(test_labels, Y_pred_test, average='weighted')))

# Compute the prediction accuracy against the training data
print("Against training set:")
print("  Accuracy:   {:.4f}".format(sklearn.metrics.accuracy_score(train_labels, Y_pred_training)))
print("  Precision:  {:.4f}".format(sklearn.metrics.precision_score(train_labels, Y_pred_training, average='weighted')))
print("  Recall:     {:.4f}".format(sklearn.metrics.recall_score(train_labels, Y_pred_training, average='weighted')))

print("--- %s seconds ---" % (time.time() - start_time))