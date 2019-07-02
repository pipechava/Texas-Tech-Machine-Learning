# Introduction to Artificial Intelligence
# Naive Bayes classifier, Part 1
# By Juan Carlos Rojas

import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load the training and test data from the Pickle file
with open("mnist_dataset_unscaled.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

# Count the unique classes
class_list = np.unique(train_labels)
num_classes = len(class_list)
num_pixels = train_data.shape[1]

# Compute the centroid image
centroid_img = np.mean(train_data, 0)
plt.figure()
plt.imshow(centroid_img.reshape(28,28), cmap="gray_r")
plt.title("Centroid")
plt.show()

# Loop through every class

prob_class_img = np.zeros( (num_classes, num_pixels) )
for classidx in range(num_classes):
  
  # Create an image of average pixels for this class
  mask = train_labels==classidx
  train_data_this_class = np.compress(mask, train_data, axis=0)

  centroid_img_class = np.mean(train_data_this_class, 0)
  plt.figure()
  plt.imshow(centroid_img_class.reshape(28,28), cmap="gray_r")
  plt.title("Centroid for Class "+str(classidx))

  # Compute probability of class for each pixel
  prob_class_img[classidx] = centroid_img_class / (centroid_img+1)
  plt.figure()
  plt.imshow(prob_class_img[classidx].reshape(28,28), cmap="gray_r")
  plt.title("Probability for class "+str(classidx))
  plt.show()


