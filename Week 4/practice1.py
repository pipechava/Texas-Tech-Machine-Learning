# Practice 1

import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load the training and test data from the Pickle file
with open("mnist_dataset_unscaled.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

# Plot a histogram of pixel values train data before standardization   
hist, bins = np.histogram(train_data, 50)
center = (bins[:-1] + bins[1:]) / 2
width = np.diff(bins)
plt.bar(center, hist, align='center', width=width)
plt.title("Hist of Train data pixel values")
plt.show()

# Plot a histogram of pixel values test data before standardization 
hist, bins = np.histogram(test_data, 50)
center = (bins[:-1] + bins[1:]) / 2
width = np.diff(bins)
plt.bar(center, hist, align='center', width=width)
plt.title("Hist of Test data pixel values")
plt.show()

# Scale the training and test data using a standard distribution
pixel_mean = np.mean(train_data)
pixel_std = np.std(train_data)
train_data = (train_data - pixel_mean) / pixel_std
test_data = (test_data - pixel_mean) / pixel_std

# Some information about the training dataset
print("Training dataset size: ", train_data.shape)
print("Class histogram: ")
print(np.histogram(train_labels, 10)[0])

# Some information about the test dataset
print("Test dataset size: ", test_data.shape)
print("Class histogram: ")
print(np.histogram(test_labels, 10)[0])

# Plot a histogram of pixel values train data after standardization 
hist, bins = np.histogram(train_data, 50)
center = (bins[:-1] + bins[1:]) / 2
width = np.diff(bins)
plt.bar(center, hist, align='center', width=width)
plt.title("Hist of Train data pixel values")
plt.show()

# Plot a histogram of pixel values test data after standardization 
hist, bins = np.histogram(test_data, 50)
center = (bins[:-1] + bins[1:]) / 2
width = np.diff(bins)
plt.bar(center, hist, align='center', width=width)
plt.title("Hist of Test data pixel values")
plt.show()

# Plot the first 5 digits as images
for idx in range(5):
  image = train_data[idx].reshape(28,28)
  plt.figure()
  plt.imshow(image, cmap="gray_r")
  plt.title("Label: "+str(train_labels[idx]))
plt.show()