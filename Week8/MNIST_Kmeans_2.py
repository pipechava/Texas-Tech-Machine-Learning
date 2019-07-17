# Introduction to Artificial Intelligence
# K-Means Clustering of MNIST dataset, part 2
# By Juan Carlos Rojas

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn.cluster
import pickle

# Load the training and test data from the Pickle file
with open("mnist_dataset_unscaled.pickle", "rb") as f:
      train_data, train_labels, _, _ = pickle.load(f)

# Scale the training data using a standard distribution
pixel_mean = np.mean(train_data)
pixel_std = np.std(train_data)
train_data = (train_data - pixel_mean) / pixel_std

# Do K-Means clustering
n_clusters = 8
print("Computing K-Means with {} clusters".format(n_clusters))
kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters, n_init=5)  
kmeans.fit(train_data)
labels = kmeans.labels_

# Compute and display the centroid image for each cluster
for idx in range(n_clusters):

    # Create a subset of the data limited to this cluster
    mask = kmeans.labels_==idx
    data_this_cluster = np.compress(mask, train_data, axis=0)

    # Plot a centroid image for this cluster
    mean_img_in_class = np.mean(data_this_cluster, 0)
    plt.figure()
    plt.imshow(mean_img_in_class.reshape(28,28), cmap="gray_r")
    plt.title("Centroid for Cluster "+str(idx))

plt.show()
