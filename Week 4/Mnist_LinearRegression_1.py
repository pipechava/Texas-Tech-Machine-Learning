# Introduction to Artificial Intelligence
# Linear regression classification of the MNIST database, part 1
# By Juan Carlos Rojas

import pickle
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn.linear_model

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

# Train a linear regression classifier
model = sklearn.linear_model.LinearRegression(fit_intercept=False)
model.fit(train_data, train_labels_onehot)

# Predict the probabilities of each class
Y_pred_proba = model.predict(test_data)

# Print the first few predicted labels
for idx in range(5):
    print("Correct: ", test_labels[idx])
    print("Predicted:")
    for classidx in range(num_classes):
        print("Class {}: {:+.3f}".format(classidx, Y_pred_proba[idx][classidx]))

# Pick the maximum
Y_pred = np.argmax(Y_pred_proba, axis=1).astype("uint8")

# Print the first few predicted labels
for idx in range(10):
    print("Predicted: ", Y_pred[idx], "Correct: ", test_labels[idx])
