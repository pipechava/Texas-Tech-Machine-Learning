# Introduction to Artificial Intelligence
# Logistic regression classification of the MNIST database, part 1
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

num_classes = len(np.unique(train_labels))

# Train a Logistic Regression  classifier
model = sklearn.linear_model.LogisticRegression(\
    multi_class = 'multinomial', solver='sag', tol=1e-2, max_iter = 50) 
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

# Explore the coefficients
print("Min coef:", np.min(model.coef_))
print("Max coef:", np.max(model.coef_))
print("Coef mean:", np.mean(model.coef_))
print("Coef stddev: ", np.std(model.coef_))

# Plot a histogram of coefficient values
hist, bins = np.histogram(model.coef_, 500)
center = (bins[:-1] + bins[1:]) / 2
width = np.diff(bins)
plt.bar(center, hist, align='center', width=width)
plt.title("Coefficient values")
plt.show()

# Display the coefficients as an image
for n in range(num_classes):
    coef_img = model.coef_[n].reshape(28, 28)
    plt.figure()
    plt.imshow(coef_img, cmap="viridis")
    plt.title("Coefficients for class "+str(n))
plt.show()
