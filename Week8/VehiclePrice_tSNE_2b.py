# Introduction to Artificial Intelligence
# t-SNE Dimensionality Reduction of Vehicle Price dataset, part 2b
# By Juan Carlos Rojas

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn.manifold
import mpl_toolkits.mplot3d
import pickle

# Load the t-SNE reduced data from the Pickle file
with open("vehicle_price_tsne_3d_5kpoints.pickle", "rb") as f:
      data_3d, = pickle.load(f)

print("Plotting 3D scatterplot")
# Plot a scatterplot of the two transformed componentes
fig = plt.figure()
ax = mpl_toolkits.mplot3d.Axes3D(fig)
ax.scatter(data_3d[:,0], data_3d[:,1], data_3d[:,2], marker='.')
plt.show()



