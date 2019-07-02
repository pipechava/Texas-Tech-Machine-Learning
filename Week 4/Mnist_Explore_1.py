# Introduction to Artificial Intelligence
# Program to explore the MNIST database, part 1
# By Juan Carlos Rojas

import numpy as np
import matplotlib.pyplot as plt

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
        data = np.array(data)
        labels = np.array(labels)
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

# Some information about the training dataset
print("Training dataset size: ", train_data.shape)
print("Class histogram: ")
print(np.histogram(train_labels, 10)[0])

# Some information about the test dataset
print("Test dataset size: ", test_data.shape)
print("Class histogram: ")
print(np.histogram(test_labels, 10)[0])

# Plot a few images
"""
for idx in range(10):
  image = train_data[idx].reshape(28,28)
  plt.figure()
  plt.imshow(image, cmap="gray_r")
  plt.title("Label: "+str(train_labels[idx]))
plt.show()
"""

# Plot a histogram of pixel values
hist, bins = np.histogram(train_data, 50)
center = (bins[:-1] + bins[1:]) / 2
width = np.diff(bins)
plt.bar(center, hist, align='center', width=width)
plt.show()

# Compute the average pixel value for all images
meanimg = np.mean(train_data, 0)
plt.figure()
plt.imshow(meanimg.reshape(28,28), cmap="gray_r")
plt.title("Mean Pixel Values Training")
plt.show()

# Compute the average pixel value for all images
meanimg = np.mean(test_data, 0)
plt.figure()
plt.imshow(meanimg.reshape(28,28), cmap="gray_r")
plt.title("Mean Pixel Values Test")
plt.show()
