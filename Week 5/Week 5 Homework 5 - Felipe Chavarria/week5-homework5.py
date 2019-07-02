#week4-homework4.py

#==============================================================================
#Practice 1
#==============================================================================

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

# Load the training and test data from the Pickle file
with open("vehicle_price_dataset_scaled.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

# Insert a column of ones
train_data["ones"] = 1
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

#==============================================================================
#Practice 2
#==============================================================================

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

# Load the training and test data from the Pickle file
with open("vehicle_price_dataset_scaled.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

# Get some lengths
ncoeffs = train_data.shape[1]
nsamples = train_data.shape[0]

# Training constants
learning_rate = .1
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

#==============================================================================
#Optional 2b
#==============================================================================

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

#==============================================================================
#Optional 2c
#==============================================================================

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

#==============================================================================
#Practice 3
#==============================================================================

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
learning_rate = 0.2
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

min_epoch = None
max_epoch = None
avg_epoch = None

epoch = 0
while epoch < n_epochs:
    
    list_cost_vals = []

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
        #if (epoch % print_step == 0):
            #print("Epoch {:4d}: Batch {:4d} Cost: {:.4f}".format(epoch, batch, cost_val))
        
        #add cost_vals to list
        list_cost_vals.append(cost_val)
        
        batch += 1
    
    # gets min, max and average cost of every mini batch in this epoch        
    max_epoch = max(list_cost_vals)
    min_epoch = min(list_cost_vals)
    avg_epoch = sum(list_cost_vals)/len(list_cost_vals)
    
    print("For Epoch {:4d}:".format(epoch))
    print("\t Min Value: ", min_epoch)
    print("\t Max Value: ", max_epoch)
    print("\t Average Value: ", avg_epoch)
    
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