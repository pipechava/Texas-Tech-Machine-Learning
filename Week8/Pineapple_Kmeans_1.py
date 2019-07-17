# Introduction to Artificial Intelligence
# K-Means Clustering of pineapple image pixels, part 1
# By Juan Carlos Rojas

import matplotlib.pyplot as plt
import skimage.io
import mpl_toolkits.mplot3d
import numpy as np
import sklearn.decomposition
import sklearn.cluster

# Load image
img = skimage.io.imread("pineapple1.png")

# Display image
plt.imshow(img)
#plt.show()

# Reshape as a vector of pixels, each with 3 color components
pixels = img[:,:,0:3].reshape(-1,3)

# Do K-Means clustering
n_clusters = 2
print("Computing K-Means with {} clusters".format(n_clusters))
kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters, n_init=5)  
kmeans.fit(pixels)
labels = kmeans.labels_

print("Computing PCA reduction to 2 dimensions")
pca = sklearn.decomposition.PCA(n_components=2)
data_2d = pca.fit_transform(pixels)

# Select a random subset
subset_len = 10000
total_len = pixels.shape[0]
reorder_map = np.random.permutation(total_len)
subset_map = reorder_map[0:subset_len]
subset = data_2d[subset_map]

# Plot a scatterplot of the two transformed components
# With color markers corresponding to the clusters
plt.figure()
plt.scatter(subset[:,0], subset[:,1], c=labels[subset_map], cmap="Set1", marker=".")
#plt.show()

# Reconstruct an image of labels
img_clusters = labels.reshape(img.shape[0], img.shape[1])
plt.figure()
plt.imshow(img_clusters, cmap="Set1")
plt.show()
