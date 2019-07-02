# Introduction to Artificial Intelligence
# Program to explore the MNIST database, part 4
# By Juan Carlos Rojas

import pickle
import numpy as np
import matplotlib.pyplot as plt

img_size = 8
filename = "mnist_dataset_{}.pickle".format(img_size)

# Load the training and test data from the Pickle file
with open(filename, "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

# Print some information about the training dataset
print("Training dataset size: ", train_data.shape)
print("Class histogram: ")
print(np.histogram(train_labels, 10)[0])

# Print some information about the test dataset
print("Test dataset size: ", test_data.shape)
print("Class histogram: ")
print(np.histogram(test_labels, 10)[0])

# Plot a histogram of pixel values
hist, bins = np.histogram(train_data, 256)
center = (bins[:-1] + bins[1:]) / 2
width = np.diff(bins)
plt.bar(center, hist, align='center', width=width)
plt.title("Train dataset pixel values")
plt.show()

# Plot a few images
#"""
for idx in range(5):
  image = train_data[idx].reshape(img_size,img_size)
  plt.figure()
  plt.imshow(image, cmap="gray_r")
  plt.title("Label: "+str(train_labels[idx]))
plt.show()
#"""


