# Introduction to Artificial Intelligence
# t-SNE Dimensionality Reduction of MNIST dataset, part 1
# By Juan Carlos Rojas

import numpy as np
import matplotlib.pyplot as plt
import sklearn.manifold
import pickle
import time

# Load the training and test data from the Pickle file
with open("mnist_dataset_unscaled.pickle", "rb") as f:
      train_data, _, _, _ = pickle.load(f)

# Pick a random subset of the data
subset_len = 5000
subset = train_data[:subset_len]

# Compute the t-SNE transformation into 2 dimensions
print("Computing t-SNE reduction into 2 dimensions")
start_time = time.time()
tsne = sklearn.manifold.TSNE(n_components=2)
data_2d = tsne.fit_transform(subset)
end_time = time.time()
print("Completed in: ", end_time - start_time);

# Plot a scatterplot of the two transformed components
plt.plot(data_2d[:,0], data_2d[:,1], ".")
plt.show()

