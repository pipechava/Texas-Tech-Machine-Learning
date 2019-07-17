#week6-homework6.py

#==============================================================================
#Practice 1
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
n_nodes_l1 = 100
batch_size = 32
learning_rate = .03
n_epochs = 100
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

# Initialize lists
cost_against_training_data = []
cost_against_test_data = []
accuracy_against_training_data = []
accuracy_against_test_data = []

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

        # Compute the average cost for all training mini-batches in this epoch
        trn_cost_avg = np.mean(cost_vals)

        # Compute the prediction accuracy against the full training data
        feed_dict = {X: train_data, Y: train_labels_onehot}
        Y_pred_training = sess.run(Y_pred_calc, feed_dict=feed_dict)
        train_accuracy = sklearn.metrics.accuracy_score(train_labels, Y_pred_training)

        # Compute the cost and prediction accuracy against the test data
        feed_dict = {X: test_data, Y: test_labels_onehot}
        test_cost = sess.run(cost, feed_dict=feed_dict)
        Y_pred_test = sess.run(Y_pred_calc, feed_dict=feed_dict)
        test_accuracy = sklearn.metrics.accuracy_score(test_labels, Y_pred_test)
      
        print("Epoch: {:4d} trn_cost: {:.5f} test_cost: {:.5f} trn_acc: {:.4f} test_acc: {:.4f}".\
              format(epoch, trn_cost_avg, test_cost, train_accuracy, test_accuracy))
        
        # Save the metrics to the history
        cost_against_training_data.append(trn_cost_avg)
        cost_against_test_data.append(test_cost)
        accuracy_against_training_data.append(train_accuracy)
        accuracy_against_test_data.append(test_accuracy)

    epoch += 1

# Accuracy, precision & recall
print("Accuracy:   {:.4f}".format(sklearn.metrics.accuracy_score(test_labels, Y_pred_test)))
print("Precision:  {:.4f}".format(sklearn.metrics.precision_score(test_labels, Y_pred_test, average='weighted')))
print("Recall:     {:.4f}".format(sklearn.metrics.recall_score(test_labels, Y_pred_test, average='weighted')))

# Compute the prediction accuracy against the training data
print("Against training set:")
print("  Accuracy:   {:.4f}".format(sklearn.metrics.accuracy_score(train_labels, Y_pred_training)))
print("  Precision:  {:.4f}".format(sklearn.metrics.precision_score(train_labels, Y_pred_training, average='weighted')))
print("  Recall:     {:.4f}".format(sklearn.metrics.recall_score(train_labels, Y_pred_training, average='weighted')))

# Print the best results (as if we had done early stopping)
epoch_hist = [i for i in range(0, n_epochs, eval_step)]

best_idx = accuracy_against_test_data.index(max(accuracy_against_test_data))
print("Max test accuracy:  {:.4f} at epoch: {}".format(accuracy_against_test_data[best_idx], epoch_hist[best_idx]))

best_idx = accuracy_against_training_data.index(max(accuracy_against_training_data))
print("Max train accuracy: {:.4f} at epoch: {}".format(accuracy_against_training_data[best_idx], epoch_hist[best_idx]))

best_idx = cost_against_test_data.index(min(cost_against_test_data))
print("Min test cost:  {:.5f} at epoch: {}".format(cost_against_test_data[best_idx], epoch_hist[best_idx]))

best_idx = cost_against_training_data.index(min(cost_against_training_data))
print("Min train cost: {:.5f} at epoch: {}".format(cost_against_training_data[best_idx], epoch_hist[best_idx]))

# Plot the metrics history
plt.plot(epoch_hist, cost_against_training_data, "b", label="Training Data")
plt.plot(epoch_hist, cost_against_test_data, label="Test Data")
plt.xlabel("epoch")
plt.ylabel("cost")
plt.title("Cost vs. epoch")
plt.figure()
plt.plot(epoch_hist, accuracy_against_training_data, "b", label="Training Data")
plt.plot(epoch_hist, accuracy_against_test_data, "r", label="Test Data")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.title("Accuracy vs. epoch")
plt.show()

#==============================================================================
#Optional 1b
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
n_nodes_l1 = 200
batch_size = 32
learning_rate = .05
n_epochs = 100
eval_step = 10

n_batches = int(np.ceil(nsamples / batch_size))

# Print the configuration
print("Batch size: {} Num batches: {} Num epochs: {} Learning rate: {}".format(batch_size, n_batches, n_epochs, learning_rate))
print("Num nodes in L1: {} Activation function: tanh".format(n_nodes_l1))

# Input vector placeholders. Length is unspecified.
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
Y = tf.placeholder(tf.float32, shape=(None, num_classes), name="Y")


############### WITH TANH ###############
# Hidden layer 1:
#   Inputs: n_inputs
#   Outputs: n_nodes_l1
#   Activation: tanh
'''
W_L1 = tf.Variable(tf.truncated_normal([n_inputs, n_nodes_l1], stddev=1/np.sqrt(n_inputs)))
b_L1 = tf.Variable(tf.zeros(n_nodes_l1))
Y_L1 = tf.nn.tanh(tf.add(tf.matmul(X, W_L1), b_L1))
'''

############### WITH RELU ###############
# Hidden layer 1:
#   Inputs: n_inputs
#   Outputs: n_nodes_l1
#   Activation: ReLU
'''
W_L1 = tf.Variable(tf.truncated_normal([n_inputs, n_nodes_l1], stddev=1/np.sqrt(n_inputs)))
b_L1 = tf.Variable(tf.zeros(n_nodes_l1))
Y_L1 = tf.nn.relu(tf.add(tf.matmul(X, W_L1), b_L1))
'''

############### WITH ELU ###############
# Hidden layer 1:
#   Inputs: n_inputs
#   Outputs: n_nodes_l1
#   Activation: ELU

W_L1 = tf.Variable(tf.truncated_normal([n_inputs, n_nodes_l1], stddev=1/np.sqrt(n_inputs)))
b_L1 = tf.Variable(tf.zeros(n_nodes_l1))
Y_L1 = tf.nn.elu(tf.add(tf.matmul(X, W_L1), b_L1))


#########################################

# Output layer:
#   Inputs: n_nodes_l1
#   Outputs: num_classes
#   Activation: softmax
W_L2 = tf.Variable(tf.truncated_normal([n_nodes_l1, num_classes], stddev=1/np.sqrt(n_nodes_l1)))
b_L2 = tf.Variable(tf.zeros(num_classes))
# softmax
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

# Initialize lists
cost_against_training_data = []
cost_against_test_data = []
accuracy_against_training_data = []
accuracy_against_test_data = []

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

        # Compute the average cost for all training mini-batches in this epoch
        trn_cost_avg = np.mean(cost_vals)

        # Compute the prediction accuracy against the full training data
        feed_dict = {X: train_data, Y: train_labels_onehot}
        Y_pred_training = sess.run(Y_pred_calc, feed_dict=feed_dict)
        train_accuracy = sklearn.metrics.accuracy_score(train_labels, Y_pred_training)

        # Compute the cost and prediction accuracy against the test data
        feed_dict = {X: test_data, Y: test_labels_onehot}
        test_cost = sess.run(cost, feed_dict=feed_dict)
        Y_pred_test = sess.run(Y_pred_calc, feed_dict=feed_dict)
        test_accuracy = sklearn.metrics.accuracy_score(test_labels, Y_pred_test)
      
        print("Epoch: {:4d} trn_cost: {:.5f} test_cost: {:.5f} trn_acc: {:.4f} test_acc: {:.4f}".\
              format(epoch, trn_cost_avg, test_cost, train_accuracy, test_accuracy))
        
        # Save the metrics to the history
        cost_against_training_data.append(trn_cost_avg)
        cost_against_test_data.append(test_cost)
        accuracy_against_training_data.append(train_accuracy)
        accuracy_against_test_data.append(test_accuracy)

    epoch += 1

# Accuracy, precision & recall
print("Accuracy:   {:.4f}".format(sklearn.metrics.accuracy_score(test_labels, Y_pred_test)))
print("Precision:  {:.4f}".format(sklearn.metrics.precision_score(test_labels, Y_pred_test, average='weighted')))
print("Recall:     {:.4f}".format(sklearn.metrics.recall_score(test_labels, Y_pred_test, average='weighted')))

# Compute the prediction accuracy against the training data
print("Against training set:")
print("  Accuracy:   {:.4f}".format(sklearn.metrics.accuracy_score(train_labels, Y_pred_training)))
print("  Precision:  {:.4f}".format(sklearn.metrics.precision_score(train_labels, Y_pred_training, average='weighted')))
print("  Recall:     {:.4f}".format(sklearn.metrics.recall_score(train_labels, Y_pred_training, average='weighted')))

# Print the best results (as if we had done early stopping)
epoch_hist = [i for i in range(0, n_epochs, eval_step)]

best_idx = accuracy_against_test_data.index(max(accuracy_against_test_data))
print("Max test accuracy:  {:.4f} at epoch: {}".format(accuracy_against_test_data[best_idx], epoch_hist[best_idx]))

best_idx = accuracy_against_training_data.index(max(accuracy_against_training_data))
print("Max train accuracy: {:.4f} at epoch: {}".format(accuracy_against_training_data[best_idx], epoch_hist[best_idx]))

best_idx = cost_against_test_data.index(min(cost_against_test_data))
print("Min test cost:  {:.5f} at epoch: {}".format(cost_against_test_data[best_idx], epoch_hist[best_idx]))

best_idx = cost_against_training_data.index(min(cost_against_training_data))
print("Min train cost: {:.5f} at epoch: {}".format(cost_against_training_data[best_idx], epoch_hist[best_idx]))

# Plot the metrics history
plt.plot(epoch_hist, cost_against_training_data, "b", label="Training Data")
plt.plot(epoch_hist, cost_against_test_data, label="Test Data")
plt.xlabel("epoch")
plt.ylabel("cost")
plt.title("Cost vs. epoch")
plt.figure()
plt.plot(epoch_hist, accuracy_against_training_data, "b", label="Training Data")
plt.plot(epoch_hist, accuracy_against_test_data, "r", label="Test Data")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.title("Accuracy vs. epoch")
plt.show()

#==============================================================================
#Practice 2
#==============================================================================

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import sklearn.metrics
import matplotlib.pyplot as plt

# Load the training and test data from the Pickle file
with open("credit_card_default_dataset.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

# Get some lengths
n_inputs = train_data.shape[1]
nsamples = train_data.shape[0]

# Training constants
n_nodes_l1_list = [5, 10, 15, 20, 30]
batch_size = 32
learning_rate = .001    # Initial rate for Adam
n_epochs = 1000
eval_step = 10

n_batches = int(np.ceil(nsamples / batch_size))

for n_nodes_l1 in n_nodes_l1_list:

    '''
    # Print the configuration
    print("Batch size: {} Num batches: {} Num epochs: {} Learning rate: {}".format(batch_size, n_batches, n_epochs, learning_rate))
    print("Num nodes in L1: {} Activation function: ELU".format(n_nodes_l1))
    '''
    # TensorFlow constants

    # Input vector placeholders. Length is unspecified.
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    Y = tf.placeholder(tf.float32, shape=(None, 1), name="Y")
    '''
    # Hidden layer 1:
    #   Inputs: n_inputs
    #   Outputs: n_nodes_l1
    #   Activation: ELU
    W_L1 = tf.Variable(tf.truncated_normal([n_inputs, n_nodes_l1], stddev=2/np.sqrt(n_inputs)))
    b_L1 = tf.Variable(tf.zeros(n_nodes_l1)) 
    Y_L1 = tf.nn.elu(tf.add(tf.matmul(X, W_L1), b_L1))
    '''
    
    # Hidden layer 1:
    #   Inputs: n_inputs
    #   Outputs: n_nodes_l1
    #   Activation: relu
    W_L1 = tf.Variable(tf.truncated_normal([n_inputs, n_nodes_l1], stddev=2/np.sqrt(n_inputs)))
    b_L1 = tf.Variable(tf.zeros(n_nodes_l1)) 
    Y_L1 = tf.nn.relu(tf.add(tf.matmul(X, W_L1), b_L1))
    

    # Output layer:
    #   Inputs: n_nodes_l1
    #   Outputs: 1
    #   Activation: logistic
    W_L2 = tf.Variable(tf.truncated_normal([n_nodes_l1, 1], stddev=1/np.sqrt(n_nodes_l1)))
    b_L2 = tf.Variable(tf.zeros(1)) 
    Y_L2_linear = tf.add(tf.matmul(Y_L1, W_L2), b_L2)

    # Cost function, plus the sigmoid part of the prediction
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits( 
            logits = Y_L2_linear, labels = Y))

    # Optimize cost through gradient descent
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    update_op = optimizer.minimize(cost)

    # Prediction probability values
    Y_pred_proba_calc = tf.nn.sigmoid(Y_L2_linear)

    # Create TensorFlow session and initialize it
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    # Initialize lists to hold the history of metrics per epoch
    trn_cost_hist = []
    test_cost_hist = []
    trn_auroc_hist = []
    test_auroc_hist = []

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

        # Evaluate and print the results so far
        if (epoch % eval_step == 0):

            # Compute the average cost for all mini-batches in this epoch
            trn_cost_avg = np.mean(cost_vals)

            # Compute the ROC AUC against the full training data
            feed_dict = {X: train_data, Y: train_labels.values.reshape(-1,1)}
            Y_pred_proba_train = sess.run(Y_pred_proba_calc, feed_dict=feed_dict)
            train_auroc = sklearn.metrics.roc_auc_score(train_labels, Y_pred_proba_train)

            # Compute the cost and ROC AUC against the test data
            feed_dict = {X: test_data, Y: test_labels.values.reshape(-1,1)}
            Y_pred_proba_test = sess.run(Y_pred_proba_calc, feed_dict=feed_dict)
            test_cost = sess.run(cost, feed_dict=feed_dict)
            test_auroc = sklearn.metrics.roc_auc_score(test_labels, Y_pred_proba_test)

            '''
            print("Epoch: {:4d} trn_cost: {:.5f} test_cost: {:.5f} trn_auroc: {:.4f} test_auroc: {:.4f}".\
                format(epoch, trn_cost_avg, test_cost, train_auroc, test_auroc))
            '''

            # Save the metrics to the history
            trn_cost_hist.append(trn_cost_avg)
            test_cost_hist.append(test_cost)
            trn_auroc_hist.append(train_auroc)
            test_auroc_hist.append(test_auroc)

        epoch += 1
    '''
    print("Batch size: {} Num batches: {} Num epochs: {} Learning rate: {}".format(batch_size, n_batches, n_epochs, learning_rate))
    print("Num nodes in L1: {} Activation function: ELU".format(n_nodes_l1))
    '''
    print("Batch size: {} Num batches: {} Num epochs: {} Learning rate: {}".format(batch_size, n_batches, n_epochs, learning_rate))
    print("Num nodes in L1: {} Activation function: RELU".format(n_nodes_l1))

    # Print the best results (as if we had done early stopping)
    epoch_hist = [i for i in range(0, n_epochs, eval_step)]

    best_idx = test_auroc_hist.index(max(test_auroc_hist))
    print("Max test ROC AUC:  {:.4f} at epoch: {}".format(test_auroc_hist[best_idx], epoch_hist[best_idx]))

    best_idx = trn_auroc_hist.index(max(trn_auroc_hist))
    print("Max train ROC AUC: {:.4f} at epoch: {}".format(trn_auroc_hist[best_idx], epoch_hist[best_idx]))

    best_idx = test_cost_hist.index(min(test_cost_hist))
    print("Min test cost:  {:.5f} at epoch: {}".format(test_cost_hist[best_idx], epoch_hist[best_idx]))

    best_idx = trn_cost_hist.index(min(trn_cost_hist))
    print("Min train cost: {:.5f} at epoch: {}".format(trn_cost_hist[best_idx], epoch_hist[best_idx]))

    '''
    # Plot the metrics history
    plt.plot(epoch_hist, trn_cost_hist, "b")
    plt.plot(epoch_hist, test_cost_hist, "r")
    plt.xlabel("epoch")
    plt.ylabel("cost")
    plt.title("Cost vs. epoch")
    plt.figure()
    plt.plot(epoch_hist, trn_auroc_hist, "b")
    plt.plot(epoch_hist, test_auroc_hist, "r")
    plt.xlabel("epoch")
    plt.ylabel("ROC AUC")
    plt.title("ROC AUC vs. epoch")
    plt.show()
    '''

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
n_nodes_l1 = 50
batch_size = 32
learning_rate = .001
n_epochs = 1000
eval_step = 10
#regularization_scale_list = [1e-6, 4e-6, 1e-5, 4e-5, 1e-4, 4e-4]
regularization_scale_list = [4e-4]

n_batches = int(np.ceil(nsamples / batch_size))

for regularization_scale in regularization_scale_list:

    # Print the configuration
    print("Batch size: {} Num batches: {} Num epochs: {} Learning rate: {}".format(batch_size, n_batches, n_epochs, learning_rate))
    print("Num nodes in L1: {} Activation function: ELU".format(n_nodes_l1))
    print("Regularization scale: {}  Regularization type: L2".format(regularization_scale))

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

    # Cost function, plus the sigmoid part of the prediction L2-Regularization
    base_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits( 
            logits = Y_L2_linear, labels = Y))
    regularization_cost = tf.reduce_sum(tf.square(W_L1)) + tf.reduce_sum(tf.square(W_L2))
    cost = regularization_cost * regularization_scale + base_cost
    '''
    # Cost function, plus the sigmoid part of the prediction L1-Regularization
    base_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits( 
            logits = Y_L2_linear, labels = Y))
    regularization_cost = tf.reduce_sum(tf.abs(W_L1)) + tf.reduce_sum(tf.abs(W_L2))
    cost = regularization_cost * regularization_scale + base_cost
    '''
    # Optimize cost through Adam
    optimizer = tf.train.AdamOptimizer(learning_rate)
    update_op = optimizer.minimize(cost)

    # Prediction values 
    Y_pred_proba = tf.nn.softmax(Y_L2_linear)
    Y_pred_calc = tf.argmax(Y_pred_proba, 1)

    # Create TensorFlow session and initialize it
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    # Initialize lists to hold the history of metrics per epoch
    trn_cost_hist = []
    test_cost_hist = []
    trn_accuracy_hist = []
    test_accuracy_hist = []

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

        # Evaluate the results every certain number of epochs
        if (epoch % eval_step == 0):

            # Compute the average cost for all mini-batches in this epoch
            trn_cost_avg = np.mean(cost_vals)

            # Compute the test cost and predictions against the full test data
            feed_dict = {X: test_data, Y: test_labels_onehot}
            test_cost, Y_pred_test = sess.run([cost, Y_pred_calc], feed_dict=feed_dict)
            test_accuracy = sklearn.metrics.accuracy_score(test_labels, Y_pred_test)

            # Run a session to compute the predictions against the full training data
            feed_dict = {X: train_data, Y: train_labels_onehot}
            Y_pred_training = sess.run(Y_pred_calc, feed_dict=feed_dict)
            train_accuracy = sklearn.metrics.accuracy_score(train_labels, Y_pred_training)
            '''
            print("Epoch: {:4d} trn_cost: {:.5f} test_cost: {:.5f} trn_acc: {:.5f} test_acc: {:.5f}"\
                .format(epoch, trn_cost_avg, test_cost, train_accuracy, test_accuracy))
            '''
            # Save the metrics to the history
            trn_cost_hist.append(trn_cost_avg)
            test_cost_hist.append(test_cost)
            trn_accuracy_hist.append(train_accuracy)
            test_accuracy_hist.append(test_accuracy)
            
        epoch += 1

    print("Epoch: {:4d} trn_cost: {:.5f} test_cost: {:.5f} trn_acc: {:.5f} test_acc: {:.5f}"\
                .format(epoch, trn_cost_avg, test_cost, train_accuracy, test_accuracy))

    # Print the best results (as if we had done early stopping)
    epoch_hist = [i for i in range(0, n_epochs, eval_step)]

    best_idx = test_accuracy_hist.index(max(test_accuracy_hist))
    print("Max test accuracy:  {:.4f} at epoch: {}".format(test_accuracy_hist[best_idx], epoch_hist[best_idx]))

    best_idx = trn_accuracy_hist.index(max(trn_accuracy_hist))
    print("Max train accuracy: {:.4f} at epoch: {}".format(trn_accuracy_hist[best_idx], epoch_hist[best_idx]))

    best_idx = test_cost_hist.index(min(test_cost_hist))
    print("Min test cost:  {:.5f} at epoch: {}".format(test_cost_hist[best_idx], epoch_hist[best_idx]))

    best_idx = trn_cost_hist.index(min(trn_cost_hist))
    print("Min train cost: {:.5f} at epoch: {}".format(trn_cost_hist[best_idx], epoch_hist[best_idx]))
    
    # Plot the metrics history
    plt.plot(epoch_hist, trn_cost_hist, "b")
    plt.plot(epoch_hist, test_cost_hist, "r")
    plt.xlabel("epoch")
    plt.ylabel("cost")
    plt.title("Cost vs. epoch")
    plt.figure()
    plt.plot(epoch_hist, trn_accuracy_hist, "b")
    plt.plot(epoch_hist, test_accuracy_hist, "r")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Accuracy vs. epoch")
    plt.show()