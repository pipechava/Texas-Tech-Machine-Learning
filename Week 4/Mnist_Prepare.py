# Introduction to Artificial Intelligence
# Program to prepare the MNIST database
# By Juan Carlos Rojas

import numpy as np
import matplotlib.pyplot as plt
import pickle

# MNIST data and label files unpacker
# From https://gist.github.com/jurebajt/5157650
# Modified by JCR to work on NumPy arrays
def get_data_and_labels(images_filename, labels_filename):
    images_file = open(images_filename, "rb")
    labels_file = open(labels_filename, "rb")
    try:
        images_file.read(4)
        num_of_items = int.from_bytes(images_file.read(4), byteorder="big")
        num_of_rows = int.from_bytes(images_file.read(4), byteorder="big")
        num_of_colums = int.from_bytes(images_file.read(4), byteorder="big")
        num_of_image_values = num_of_rows * num_of_colums
        print("Number of images: ", num_of_items)
        print("Size of images: ", num_of_rows, "by", num_of_colums)
        labels_file.read(8)
        data = [[None for x in range(num_of_image_values)]
                for y in range(num_of_items)]
        labels = []
        for item in range(num_of_items):
            if (item % 1000) == 0:      
                  print("Current image number: %7d" % item)
            for value in range(num_of_image_values):
                data[item][value] = int.from_bytes(images_file.read(1),
                                                   byteorder="big")
            labels.append(int.from_bytes(labels_file.read(1), byteorder="big"))
        # Convert to NumPy arrays
        data = np.array(data).astype(np.uint8)
        labels = np.array(labels).astype(np.uint8)
        return data, labels
    finally:
        images_file.close()
        labels_file.close()
        print("Complete")

# Read the train data and labels files
print("Reading training dataset")
train_data, train_labels = get_data_and_labels("train-images.idx3-ubyte", "train-labels.idx1-ubyte")
train_size = train_data.shape[0]

print("Reading test dataset")
test_data, test_labels = get_data_and_labels("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")
test_size = test_data.shape[0]

# Plot a histogram of pixel values before normalization
"""
hist, bins = np.histogram(train_data, 50)
center = (bins[:-1] + bins[1:]) / 2
width = np.diff(bins)
plt.bar(center, hist, align='center', width=width)
plt.show()
"""

# Compute the mean and stddev
pixel_mean = np.mean(train_data)
pixel_std = np.std(train_data)
print("Pixel mean:", pixel_mean)
print("Pixel stddev:", pixel_std)

# Scale the training and test data
train_data_scaled = (train_data - pixel_mean) / pixel_std
test_data_scaled = (test_data - pixel_mean) / pixel_std

# Plot a histogram of pixel values
hist, bins = np.histogram(train_data_scaled, 50)
center = (bins[:-1] + bins[1:]) / 2
width = np.diff(bins)
plt.bar(center, hist, align='center', width=width)
plt.title("Train dataset scaled")
plt.show()

hist, bins = np.histogram(test_data_scaled, 50)
center = (bins[:-1] + bins[1:]) / 2
width = np.diff(bins)
plt.bar(center, hist, align='center', width=width)
plt.title("Test dataset scaled")
plt.show()

# Archive the scaled training and test datasets into a pickle file
#with open("mnist_dataset.pickle", 'wb') as f:
#      pickle.dump([train_data_scaled, train_labels, test_data_scaled, test_labels], \
#                  f, pickle.HIGHEST_PROTOCOL)

# Archive the original set as uint8, to save space
with open("mnist_dataset_unscaled.pickle", 'wb') as f:
      pickle.dump([train_data, train_labels, test_data, test_labels], \
                  f, pickle.HIGHEST_PROTOCOL)

