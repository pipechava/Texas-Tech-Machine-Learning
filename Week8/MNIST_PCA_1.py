# Introduction to Artificial Intelligence
# PCA Dimensionality Reduction of MNIST dataset, part 1
# By Juan Carlos Rojas

import numpy as np
import matplotlib.pyplot as plt
import pickle
import sklearn.decomposition

# Load the training and test data from the Pickle file
with open("mnist_dataset_unscaled.pickle", "rb") as f:
      train_data, _, _, _ = pickle.load(f)

# Compute the PCA transformation into 2 dimensions
print("Computing PCA reduction")
pca = sklearn.decomposition.PCA(n_components=2)
data_2d = pca.fit_transform(train_data)

# Pick a random subset of the results
subset_len = 5000
subset = data_2d[:subset_len]

# Plot a scatterplot of the two transformed components
plt.plot(subset[:,0], subset[:,1], ".")
plt.show()

