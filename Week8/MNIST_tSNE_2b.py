# Introduction to Artificial Intelligence
# t-SNE Dimensionality Reduction of MNIST dataset, part 2b
# By Juan Carlos Rojas

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import sklearn.manifold
import pickle
import time

# Load the t-SNE reduced data from the Pickle file
with open("mnist_tsne_3d_5kpoints.pickle", "rb") as f:
      data_3d, = pickle.load(f)

print("Plotting 3D scatterplot")
# Plot a scatterplot of the transformed componentes
fig = plt.figure()
ax = mpl_toolkits.mplot3d.Axes3D(fig)
ax.scatter(data_3d[:,0], data_3d[:,1], data_3d[:,2], marker='.')
plt.show()

