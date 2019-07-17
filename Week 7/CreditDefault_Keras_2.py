# Introduction to Artificial Intelligence
# Neural network classifier for credit default using Keras, part 1
# By Juan Carlos Rojas

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


# Load the training and test data from the Pickle file
with open("credit_card_default_dataset.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

# Get some lengths
n_inputs = train_data.shape[1]
nsamples = train_data.shape[0]

# Training constants
n_nodes = 300
n_layers = 3
batch_size = 64
n_epochs = 1000
dropout_rate = 0.5

n_nodes_per_layer = n_nodes // n_layers
n_batches = int(np.ceil(nsamples / batch_size))

# Print the configuration
print("Batch size: {} Num batches: {} Num epochs: {} ".format(batch_size, n_batches, n_epochs))
print("Num layers: {}  total num nodes: {}".format(n_layers, n_nodes))
print("Dropout rate: {}".format(dropout_rate))

# Custom metric function to compute the ROC AUC score
def auroc(y_true, y_pred):
    return tf.py_func(sklearn.metrics.roc_auc_score, (y_true, y_pred), tf.double)

#
# Keras definitions
#

# Create a neural network model
model = keras.models.Sequential()

# First layer (need to specify the input size)
print("Adding layer with {} nodes".format(n_nodes_per_layer))
model.add(keras.layers.Dense( n_nodes_per_layer, input_shape=(n_inputs,),
        activation='elu',
        kernel_initializer='he_normal', bias_initializer='zeros'))
model.add(keras.layers.Dropout(dropout_rate))

# Other hidden layers
for n in range(1, n_layers):
    print("Adding layer with {} nodes".format(n_nodes_per_layer))
    model.add(keras.layers.Dense( n_nodes_per_layer, 
        activation='elu',
        kernel_initializer='he_normal', bias_initializer='zeros'))
    model.add(keras.layers.Dropout(dropout_rate))


# Output layer
print("Adding output layer with 1 node")
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
        verbose=2,
        )
end_time = time.time()
print("Training time: ", end_time - start_time);

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

# Find the best costs & metrics
test_auroc_hist = history.history['val_auroc']
best_idx = test_auroc_hist.index(max(test_auroc_hist))
print("Max test AUC of ROC:  {:.4f} at epoch: {}".format(test_auroc_hist[best_idx], best_idx))

trn_auroc_hist = history.history['auroc']
best_idx = trn_auroc_hist.index(max(trn_auroc_hist))
print("Max train AUC of ROC: {:.4f} at epoch: {}".format(trn_auroc_hist[best_idx], best_idx))

test_cost_hist = history.history['val_loss']
best_idx = test_cost_hist.index(min(test_cost_hist))
print("Min test cost:  {:.5f} at epoch: {}".format(test_cost_hist[best_idx], best_idx))

trn_cost_hist = history.history['loss']
best_idx = trn_cost_hist.index(min(trn_cost_hist))
print("Min train cost: {:.5f} at epoch: {}".format(trn_cost_hist[best_idx], best_idx))

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
