# Introduction to Artificial Intelligence
# PCA Dimensionality Reduction of pineapple image pixels, part 1
# By Juan Carlos Rojas

import matplotlib.pyplot as plt
import skimage.io
import mpl_toolkits.mplot3d
import numpy as np
import sklearn.decomposition

# Load image
img = skimage.io.imread("pineapple1.png")

# Reshape as a vector of pixels, each with 3 color components
pixels = img[:,:,0:3].reshape(-1,3)

# Select a random subset
subset_len = 5000
total_len = pixels.shape[0]
reorder_map = np.random.permutation(total_len)
subset_map = reorder_map[0:subset_len]
pixels_subset = pixels[subset_map]

print("Computing PCA reduction")
pca = sklearn.decomposition.PCA(n_components=2)
data_2d = pca.fit_transform(pixels_subset)

# Plot a scatterplot of the two transformed components
plt.plot(data_2d[:,0], data_2d[:,1], ".")
plt.show()

