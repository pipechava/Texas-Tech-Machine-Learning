# Introduction to Artificial Intelligence
# Program to explore the MNIST database, part 3
# By Juan Carlos Rojas

import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load the training and test data from the Pickle file
with open("mnist_dataset_unscaled.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

# Scale the training and test data using a standard distribution
pixel_mean = np.mean(train_data)
pixel_std = np.std(train_data)
train_data = (train_data - pixel_mean) / pixel_std
test_data = (test_data - pixel_mean) / pixel_std

# Count the unique classes
class_list = np.unique(train_labels)
num_classes = len(class_list)

# Compute the centroid image (the mean pixel value)
centroid_img = np.mean(train_data, 0)
plt.figure()
plt.imshow(centroid_img.reshape(28,28), cmap="gray_r")
plt.title("Centroid Training")
plt.show()

# Compute the average pixel value for all images
centroid_img_test = np.mean(test_data, 0)
plt.figure()
plt.imshow(centroid_img_test.reshape(28,28), cmap="gray_r")
plt.title("Centroid Test")
plt.show()

# Compute an average image per class
for classidx in range(num_classes):
  
  # Create an image of average pixels for this class
  mask = train_labels==classidx
  train_data_this_class = np.compress(mask, train_data, axis=0)

  mean_img_in_class = np.mean(train_data_this_class, 0)
  plt.figure()
  plt.imshow(mean_img_in_class.reshape(28,28), cmap="gray_r")
  plt.title("Centroid for Class "+str(classidx))

  plt.show()

