# Introduction to Artificial Intelligence
# Neural Network regression of vehicle price in Keras, part 2
# By Juan Carlos Rojas

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import keras.models
import keras.layers
import keras.utils
import keras.regularizers
import keras.backend as K
import matplotlib.pyplot as plt
import time

# Load the training and test data from the Pickle file
with open("vehicle_price_dataset_scaled.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

# Get some lengths
n_inputs = train_data.shape[1]
nsamples = train_data.shape[0]

# Training constants
n_nodes = 1000
n_layers = 3
batch_size = 32
n_epochs = 500
dropout_rate = 0.10

n_nodes_per_layer = n_nodes // n_layers
n_batches = int(np.ceil(nsamples / batch_size))

# Print the configuration
print("Batch size: {} Num batches: {} Num epochs: {}".format(batch_size, n_batches, n_epochs))
print("Num layers: {}  total num nodes: {}".format(n_layers, n_nodes))
print("Dropout rate: {}".format(dropout_rate))

# Define custom metric function for RMSE
def rmse (y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred -y_true)))

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
        activation='linear',
        kernel_initializer='glorot_normal', bias_initializer='zeros'))

# Define the optimizer
optimizer = keras.optimizers.Adam(lr=0.001)
#optimizer = keras.optimizers.Adadelta(lr=1.0)
#optimizer = keras.optimizers.Adagrad(lr=0.01)

# Define cost function and optimization strategy
model.compile(
        optimizer=optimizer,
        loss='mean_squared_error', 
        metrics=[rmse]    
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


# Get the final prediction errors
"""
_, rmse_test = model.evaluate(test_data, test_labels, batch_size=None, verbose=0)
_, rmse_train = model.evaluate(train_data, train_labels, batch_size=None, verbose=0)

print("Test RMSE:     {:.1f}".format(rmse_test))
print("Training RMSE: {:.1f}".format(rmse_train))
"""

# Find the best costs & metrics
test_rmse_hist = history.history['val_rmse']
best_idx = test_rmse_hist.index(min(test_rmse_hist))
print("Min test RMSE:  {:.1f} at epoch: {}".format(test_rmse_hist[best_idx], best_idx))

trn_rmse_hist = history.history['rmse']
best_idx = trn_rmse_hist.index(min(trn_rmse_hist))
print("Min train RMSE: {:.1f} at epoch: {}".format(trn_rmse_hist[best_idx], best_idx))

test_cost_hist = history.history['val_loss']
best_idx = test_cost_hist.index(min(test_cost_hist))
print("Min test cost:  {:.0f} at epoch: {}".format(test_cost_hist[best_idx], best_idx))

trn_cost_hist = history.history['loss']
best_idx = trn_cost_hist.index(min(trn_cost_hist))
print("Min train cost: {:.0f} at epoch: {}".format(trn_cost_hist[best_idx], best_idx))


# Plot the history of the loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Cost')
plt.ylabel('Cost')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')

# Plot the history of the metric
plt.figure()
plt.plot(history.history['rmse'])
plt.plot(history.history['val_rmse'])
plt.title('Model RMSE')
plt.ylabel('RMSE')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
