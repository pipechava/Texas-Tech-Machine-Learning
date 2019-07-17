#week7-homework7.py

#==============================================================================
#Practice 1
#==============================================================================

import pickle
import numpy as np
import sklearn.metrics
import keras.models
import keras.layers
import keras.utils
import keras.regularizers
import matplotlib.pyplot as plt

# Load the training and test data from the Pickle file
with open("mnist_dataset_unscaled.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

# Scale the training and test data
pixel_mean = np.mean(train_data)
pixel_std = np.std(train_data)
train_data = (train_data - pixel_mean) / pixel_std
test_data = (test_data - pixel_mean) / pixel_std

# One-hot encode the labels
train_labels_onehot = keras.utils.to_categorical(train_labels)
test_labels_onehot = keras.utils.to_categorical(test_labels)
num_classes = test_labels_onehot.shape[1]

# Get some lengths
n_inputs = train_data.shape[1]
nsamples = train_data.shape[0]

# Training constants
n_nodes_l1 = 200
batch_size = 32
n_epochs = 100

n_batches = int(np.ceil(nsamples / batch_size))

# Print the configuration
print("Batch size: {} Num batches: {} Num epochs: {}".format(batch_size, n_batches, n_epochs))
print("Num nodes in L1: {} Activation function: ELU".format(n_nodes_l1))
print("No regularization")

#
# Keras definitions
#

# Create a neural network model
model = keras.models.Sequential()

# Hidden layer 1:
#   Inputs: n_inputs
#   Outputs: n_nodes_l1
#   Activation: ELU
model.add(keras.layers.Dense(
        n_nodes_l1,
        input_shape=(n_inputs,),
        activation='elu',
        kernel_initializer='he_normal',
        bias_initializer='zeros',
        ))

# Output layer:
#   Inputs: n_nodes_l1
#   Outputs: num_classes
#   Activation: softmax
model.add(keras.layers.Dense(
        num_classes,
        activation='softmax',
        kernel_initializer='glorot_normal',
        bias_initializer='zeros',
        ))

# Define the optimizer
#optimizer = keras.optimizers.Adam(lr=0.001)
optimizer = keras.optimizers.RMSprop(lr=0.001)
#optimizer = keras.optimizers.Adadelta(lr=1.0)
#optimizer = keras.optimizers.Adagrad(lr=0.01)
        
# Define cost function and optimization strategy
model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
        )

# Train the neural network
history = model.fit(
        train_data,
        train_labels_onehot,
        epochs=n_epochs,
        batch_size=batch_size,
        validation_data=(test_data, test_labels_onehot),
        verbose=2,
        )

"""
# Get the final prediction values
Y_pred_test = model.predict_classes(test_data, batch_size=None)

# Accuracy, precision & recall
print("Accuracy:   {:.4f}".format(sklearn.metrics.accuracy_score(test_labels, Y_pred_test)))
print("Precision:  {:.4f}".format(sklearn.metrics.precision_score(test_labels, Y_pred_test, average='weighted')))
print("Recall:     {:.4f}".format(sklearn.metrics.recall_score(test_labels, Y_pred_test, average='weighted')))
"""

# Find the best costs & metrics
test_accuracy_hist = history.history['val_acc']
best_idx = test_accuracy_hist.index(max(test_accuracy_hist))
print("Max test accuracy:  {:.4f} at epoch: {}".format(test_accuracy_hist[best_idx], best_idx))

trn_accuracy_hist = history.history['acc']
best_idx = trn_accuracy_hist.index(max(trn_accuracy_hist))
print("Max train accuracy: {:.4f} at epoch: {}".format(trn_accuracy_hist[best_idx], best_idx))

test_cost_hist = history.history['val_loss']
best_idx = test_cost_hist.index(min(test_cost_hist))
print("Min test cost:  {:.5f} at epoch: {}".format(test_cost_hist[best_idx], best_idx))

trn_cost_hist = history.history['loss']
best_idx = trn_cost_hist.index(min(trn_cost_hist))
print("Min train cost: {:.5f} at epoch: {}".format(trn_cost_hist[best_idx], best_idx))

'''
# Plot the history of the cost
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Cost')
plt.ylabel('Cost')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')

# Plot the history of the metric
plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
'''

'''
Would you recommend using RMSprop for this problem?
    No, it didn't gave that bad of results since it was arround the 98% of accuracy, but overall AdaDelta gave slightly better results.
'''

#==============================================================================
#Practice 2
#==============================================================================

import pickle
import numpy as np
import sklearn.metrics
import keras.models
import keras.layers
import keras.utils
import keras.regularizers
import matplotlib.pyplot as plt

# Load the training and test data from the Pickle file
with open("mnist_dataset_unscaled.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

# Scale the training and test data
pixel_mean = np.mean(train_data)
pixel_std = np.std(train_data)
train_data = (train_data - pixel_mean) / pixel_std
test_data = (test_data - pixel_mean) / pixel_std

# One-hot encode the labels
train_labels_onehot = keras.utils.to_categorical(train_labels)
test_labels_onehot = keras.utils.to_categorical(test_labels)
num_classes = test_labels_onehot.shape[1]

# Get some lengths
n_inputs = train_data.shape[1]
nsamples = train_data.shape[0]

# Training constants
total_nodes = 300
n_nodes_l1 = round(total_nodes * 0.3)
n_nodes_l2 = total_nodes - n_nodes_l1
batch_size = 32
n_epochs = 100
regularization_scale = 0

n_batches = int(np.ceil(nsamples / batch_size))

# Print the configuration
print("Batch size: {} Num batches: {} Num epochs: {}".format(batch_size, n_batches, n_epochs))
print("Num nodes in L1: {} L2: {}".format(n_nodes_l1, n_nodes_l2))
print("Regularization scale: {}".format(regularization_scale))

#
# Keras definitions
#

# Create a neural network model
model = keras.models.Sequential()

# Hidden layer 1
model.add(keras.layers.Dense( n_nodes_l1, input_shape=(n_inputs,),
        activation='elu',
        kernel_initializer='he_normal', bias_initializer='zeros',
        kernel_regularizer=keras.regularizers.l2(regularization_scale)))

model.add(keras.layers.Dense( n_nodes_l2, 
        activation='elu',
        kernel_initializer='he_normal', bias_initializer='zeros',
        kernel_regularizer=keras.regularizers.l2(regularization_scale)))

# Output layer
model.add(keras.layers.Dense(num_classes,
        activation='softmax',
        kernel_initializer='glorot_normal', bias_initializer='zeros',
        kernel_regularizer=keras.regularizers.l2(regularization_scale)))

# Define the optimizer
#optimizer = keras.optimizers.Adam(lr=0.001)
optimizer = keras.optimizers.Adadelta(lr=1.0)
#optimizer = keras.optimizers.Adagrad(lr=0.01)
        
# Define cost function and optimization strategy
model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
        )

# Train the neural network
history = model.fit(
        train_data,
        train_labels_onehot,
        epochs=n_epochs,
        batch_size=batch_size,
        validation_data=(test_data, test_labels_onehot),
        verbose=2,
        )

# Find the best costs & metrics
test_accuracy_hist = history.history['val_acc']
best_idx = test_accuracy_hist.index(max(test_accuracy_hist))
print("Max test accuracy:  {:.4f} at epoch: {}".format(test_accuracy_hist[best_idx], best_idx))

trn_accuracy_hist = history.history['acc']
best_idx = trn_accuracy_hist.index(max(trn_accuracy_hist))
print("Max train accuracy: {:.4f} at epoch: {}".format(trn_accuracy_hist[best_idx], best_idx))

test_cost_hist = history.history['val_loss']
best_idx = test_cost_hist.index(min(test_cost_hist))
print("Min test cost:  {:.5f} at epoch: {}".format(test_cost_hist[best_idx], best_idx))

trn_cost_hist = history.history['loss']
best_idx = trn_cost_hist.index(min(trn_cost_hist))
print("Min train cost: {:.5f} at epoch: {}".format(trn_cost_hist[best_idx], best_idx))

'''
# Plot the history of the cost
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Cost')
plt.ylabel('Cost')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')

# Plot the history of the metric
plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
'''

#==============================================================================
#Practice 3
#==============================================================================

import numpy as np
import pandas as pd
import pickle
import sklearn.metrics
import keras.models
import keras.layers
import keras.utils
import keras.regularizers
import tensorflow as tf
import matplotlib.pyplot as plt
import time

tf.logging.set_verbosity(tf.logging.FATAL)

# Load the training and test data from the Pickle file
with open("credit_card_default_dataset.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

# Get some lengths
n_inputs = train_data.shape[1]
nsamples = train_data.shape[0]

dropout_rate_list = [0.1, 0.3, 0.5, 0.7, 0.9]

for DO_rate in dropout_rate_list:      
    # Training constants
    n_nodes = 1000
    n_layers = 3
    batch_size = 64
    n_epochs = 2000
    dropout_rate = DO_rate

    n_nodes_per_layer = n_nodes // n_layers
    n_batches = int(np.ceil(nsamples / batch_size))

    # Custom metric function to compute the ROC AUC score
    def auroc(y_true, y_pred):
        return tf.py_func(sklearn.metrics.roc_auc_score, (y_true, y_pred), tf.double)

    #
    # Keras definitions
    #

    # Create a neural network model
    model = keras.models.Sequential()

    # First layer (need to specify the input size)
    #print("Adding layer with {} nodes".format(n_nodes_per_layer))
    model.add(keras.layers.Dense( n_nodes_per_layer, input_shape=(n_inputs,),
            activation='elu',
            kernel_initializer='he_normal', bias_initializer='zeros'))
    model.add(keras.layers.Dropout(dropout_rate))

    # Other hidden layers
    for n in range(1, n_layers):
        #print("Adding layer with {} nodes".format(n_nodes_per_layer))
        model.add(keras.layers.Dense( n_nodes_per_layer, 
            activation='elu',
            kernel_initializer='he_normal', bias_initializer='zeros'))
        model.add(keras.layers.Dropout(dropout_rate))


    # Output layer
    #print("Adding output layer with 1 node")
    model.add(keras.layers.Dense(1,
            activation='sigmoid',
            kernel_initializer='glorot_normal', bias_initializer='zeros'))

    # Define the optimizer
    optimizer = keras.optimizers.Adam(lr=0.001)
    #optimizer = keras.optimizers.Adadelta(lr=1.0)
    #optimizer = keras.optimizers.Adagrad(lr=0.01)

    # Define cost function and optimization strategy
    model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy', # Logistic cross entropy
            metrics=[auroc]    
            )

    # Train the neural network
    start_time = time.time()
    history = model.fit(
            train_data,
            train_labels,
            epochs=n_epochs,
            batch_size=batch_size,
            validation_data=(test_data, test_labels),
            verbose=0,
            )
    

    """
    # Get the prediction values
    Y_pred_test = model.predict(test_data, batch_size=None)
    Y_pred_training = model.predict(train_data, batch_size=None)

    auc_score = sklearn.metrics.roc_auc_score(test_labels, Y_pred_test)
    print("Test AUC score: {:.4f}".format(auc_score))

    # Predict new labels for training data
    auc_score_training = sklearn.metrics.roc_auc_score(train_labels, Y_pred_training)
    print("Training AUC score: {:.4f}".format(auc_score_training))
    """

    print("\n")

    # Print the configuration
    print("Batch size: {} Num batches: {} Num epochs: {} ".format(batch_size, n_batches, n_epochs))
    print("Num layers: {}  total num nodes: {}".format(n_layers, n_nodes))
    print("Dropout rate: {}".format(dropout_rate))

    # Find the best costs & metrics
    test_auroc_hist = history.history['val_auroc']
    best_idx = test_auroc_hist.index(max(test_auroc_hist))
    print("     Max test AUC of ROC:  {:.4f} at epoch: {}".format(test_auroc_hist[best_idx], best_idx))

    trn_auroc_hist = history.history['auroc']
    best_idx = trn_auroc_hist.index(max(trn_auroc_hist))
    print("     Max train AUC of ROC: {:.4f} at epoch: {}".format(trn_auroc_hist[best_idx], best_idx))

    test_cost_hist = history.history['val_loss']
    best_idx = test_cost_hist.index(min(test_cost_hist))
    print("     Min test cost:  {:.5f} at epoch: {}".format(test_cost_hist[best_idx], best_idx))

    trn_cost_hist = history.history['loss']
    best_idx = trn_cost_hist.index(min(trn_cost_hist))
    print("     Min train cost: {:.5f} at epoch: {}".format(trn_cost_hist[best_idx], best_idx))

    end_time = time.time()
    print("Training time: ", end_time - start_time)

    print("\n")

    '''
    # Plot the history of the loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Cost')
    plt.ylabel('Cost')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # Plot the history of the metric
    plt.figure()
    plt.plot(history.history['auroc'])
    plt.plot(history.history['val_auroc'])
    plt.title('Model AUC of ROC')
    plt.ylabel('AUC of ROC')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    '''