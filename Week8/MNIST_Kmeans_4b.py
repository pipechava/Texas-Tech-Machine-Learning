# Introduction to Artificial Intelligence
# K-Means Clustering of MNIST dataset, part 4
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

avg_score_list=[]

min_k = 2
max_k = 30
k_step = 1

for n_clusters in range(min_k, max_k, k_step):

    # Do K-Means clustering
    print("Computing K-Means with {} clusters".format(n_clusters))
    kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters, n_init=5)  
    kmeans.fit(train_data)
    labels = kmeans.labels_

    # Compute the average silhouette score
    #print("Computing silhouette scores")

    # Pick a random subset of the data
    subset_len = 10000

    total_len = train_data.shape[0]
    reorder_map = np.random.permutation(total_len)
    subset_map = reorder_map[0:subset_len]

    # Compute the silhouette scores for each sample
    silhouette_values = sklearn.metrics.silhouette_samples(train_data[subset_map], labels[subset_map])

    # For each cluster, compute the mean of its silhouette scores
    for idx in range(n_clusters):

        # Create a subset of the data limited to this cluster
        silhouette_values_this_cluster = silhouette_values[labels[subset_map]==idx]

        # Print the average score for this cluster
        #print("Cluster {}: Avg. Silhouette score: {:.3f}".format(idx, silhouette_values_this_cluster.mean()))

    print("  Avg. silhouette_score: {:.3f}".format(silhouette_values.mean()))
    avg_score_list.append(silhouette_values.mean())

k_list = [x for x in range(min_k, max_k, k_step)]
plt.plot(k_list, avg_score_list)
plt.xlabel("k")
plt.ylabel("Avg. Silhouette Score")
plt.show()

