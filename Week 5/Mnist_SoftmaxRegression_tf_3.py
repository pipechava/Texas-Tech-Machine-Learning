# Introduction to Artificial Intelligence
# Softmax regression classification of the MNIST database in TensorFlow, part 3
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
batch_size = 6000
learning_rate = .2
n_epochs = 1000
print_step = 10

n_batches = int(np.ceil(nsamples / batch_size))

# Print the configuration
print("Batch size: {} Num batches: {} Learning rate: {}".format(batch_size, n_batches, learning_rate))

# Input vector placeholders. Length is unspecified.
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
Y = tf.placeholder(tf.float32, shape=(None, num_classes), name="Y")

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

# Prediction values 
Y_pred_proba = tf.nn.softmax(Y_linear)
Y_pred_calc = tf.argmax(Y_pred_proba, 1)

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
        Y_batch = train_labels_onehot[dataidx:(dataidx+batch_size)]
        feed_dict = {X: X_batch, Y: Y_batch}

        # Run one iteration of the computation session to update coefficients
        _, cost_val = sess.run([update_op, cost], feed_dict=feed_dict)

        # Print the cost for each batch in some epochs
        if (epoch % print_step == 0):
            print("Epoch {:4d}: Batch {:4d} Cost: {:.4f}".format(epoch, batch, cost_val))
            
        batch += 1
    epoch += 1

# Run a session to compute the predictions against the training data
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

