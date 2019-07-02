# Introduction to Artificial Intelligence
# Random forest classification of the MNIST database, part 1
# By Juan Carlos Rojas

import pickle
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn.ensemble

# Load the training and test data from the Pickle file
with open("mnist_dataset_unscaled.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

# Scale the training and test data
pixel_mean = np.mean(train_data)
pixel_std = np.std(train_data)
train_data = (train_data - pixel_mean) / pixel_std
test_data = (test_data - pixel_mean) / pixel_std

num_classes = len(np.unique(train_labels))

# Train a Decision Tree classifier
model = sklearn.ensemble.RandomForestClassifier(\
    n_estimators = 100,
    min_samples_leaf = 5) 
model.fit(train_data, train_labels)

# Predict the labels for all the test cases
Y_pred = model.predict(test_data)

# Confusion matrix
cmatrix = sklearn.metrics.confusion_matrix(test_labels, Y_pred)
print("Confusion Matrix:")
print(cmatrix)

# Accuracy, precision & recall
print("Accuracy:   {:.3f}".format(sklearn.metrics.accuracy_score(test_labels, Y_pred)))
print("Precision:  {:.3f}".format(sklearn.metrics.precision_score(test_labels, Y_pred, average='weighted')))
print("Recall:     {:.3f}".format(sklearn.metrics.recall_score(test_labels, Y_pred, average='weighted')))

# Per-Class Precision & Recall
precision = sklearn.metrics.precision_score(test_labels, Y_pred, average=None)
recall = sklearn.metrics.recall_score(test_labels, Y_pred, average=None)
for n in range(num_classes):
    print("  Class {}: Precision: {:.3f} Recall: {:.3f}".format(n, precision[n], recall[n]))

# Compute the prediction accuracy against the training data
print("Against training set:")
Y_pred_training = model.predict(train_data)
print("  Accuracy:   {:.3f}".format(sklearn.metrics.accuracy_score(train_labels, Y_pred_training)))
print("  Precision:  {:.3f}".format(sklearn.metrics.precision_score(train_labels, Y_pred_training, average='weighted')))
print("  Recall:     {:.3f}".format(sklearn.metrics.recall_score(train_labels, Y_pred_training, average='weighted')))

# Plot some of the incorrect predictions
#"""
num_displayed = 0
x = 0
while (num_displayed < 10):
    x += 1

    # Skip correctly predicted 
    if (Y_pred[x] == test_labels[x]):
        continue

    num_displayed += 1

    # Display the images
    image = test_data[x].reshape(28,28)
    plt.figure()
    plt.imshow(image, cmap="gray_r")
    plt.title("Predicted: "+str(Y_pred[x])+" Correct: "+str(test_labels[x]))
    plt.show()
#"""

