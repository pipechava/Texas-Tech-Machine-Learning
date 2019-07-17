# Introduction to Artificial Intelligence
# K-Means Clustering of MNIST dataset, part 1
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

# Visualize the clustering results on the t-SNR reduced visualization
subset_len = 5000

print("Computing t-SNE reduction into 2 dimensions")
tsne = sklearn.manifold.TSNE(n_components=2)
data_2d = tsne.fit_transform(train_data[:subset_len])

# Plot a scatterplot of the two transformed components
# With color markers corresponding to the clusters
plt.scatter(data_2d[:,0], data_2d[:,1], c=labels[:subset_len], cmap="Set1", marker=".")
plt.show()

