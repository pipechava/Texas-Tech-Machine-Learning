# Introduction to Artificial Intelligence
# Logistic regression classification of the MNIST database in TensorFlow, part 1
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
ncoeffs = train_data.shape[1]
nsamples = train_data.shape[0]

# Training constants
learning_rate = .3
n_iterations = 10000
print_step = 100

# TensorFlow constants

# Input data matrices
X = tf.constant(train_data.astype(np.float32))
Y = tf.constant(train_labels_onehot.astype(np.float32))
X_test = tf.constant(test_data.astype(np.float32))

# Initial coefficients & bias

# Start with uniformly random values
W = tf.Variable(tf.random_uniform([ncoeffs, num_classes], -1.0, 1.0))

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
        print("iteration {:4d}:  Cost: {}".format(iteration, cost_val))
    iteration += 1

# Run a session to compute the predictions against the training data
Y_pred_training = sess.run(Y_pred_training_calc)

# Run a session to compute the predictions against the test data
Y_pred_test = sess.run(Y_pred_test_calc)

# Confusion matrix
cmatrix = sklearn.metrics.confusion_matrix(test_labels, Y_pred_test)
print("Confusion Matrix:")
print(cmatrix)

# Accuracy, precision & recall
print("Accuracy:   {:.3f}".format(sklearn.metrics.accuracy_score(test_labels, Y_pred_test)))
print("Precision:  {:.3f}".format(sklearn.metrics.precision_score(test_labels, Y_pred_test, average='weighted')))
print("Recall:     {:.3f}".format(sklearn.metrics.recall_score(test_labels, Y_pred_test, average='weighted')))

# Per-Class Precision & Recall
precision = sklearn.metrics.precision_score(test_labels, Y_pred_test, average=None)
recall = sklearn.metrics.recall_score(test_labels, Y_pred_test, average=None)
for n in range(num_classes):
    print("  Class {}: Precision: {:.3f} Recall: {:.3f}".format(n, precision[n], recall[n]))

# Compute the prediction accuracy against the training data
print("Against training set:")
print("  Accuracy:   {:.3f}".format(sklearn.metrics.accuracy_score(train_labels, Y_pred_training)))
print("  Precision:  {:.3f}".format(sklearn.metrics.precision_score(train_labels, Y_pred_training, average='weighted')))
print("  Recall:     {:.3f}".format(sklearn.metrics.recall_score(train_labels, Y_pred_training, average='weighted')))

