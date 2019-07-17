# Introduction to Artificial Intelligence
# t-SNE Dimensionality Reduction of MNIST dataset, part 2
# By Juan Carlos Rojas

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import sklearn.manifold
import pickle
import time

# Load the training and test data from the Pickle file
with open("mnist_dataset_unscaled.pickle", "rb") as f:
      train_data, train_labels, _, _ = pickle.load(f)

# Pick a random subset of the data
subset_len = 5000
subset = train_data[:subset_len]

print("Computing t-SNE reduction into 3 dimensions")
start_time = time.time()
tsne = sklearn.manifold.TSNE(n_components=3)
data_3d = tsne.fit_transform(train_data[:subset_len])
end_time = time.time()
print("Completed in: ", end_time - start_time);

print("Plotting 3D scatterplot")
# Plot a scatterplot of the transformed componentes
fig = plt.figure()
ax = mpl_toolkits.mplot3d.Axes3D(fig)
ax.scatter(data_3d[:,0], data_3d[:,1], data_3d[:,2], marker='.')
plt.show()

# Archive the t-SNE 3D projection data
with open("mnist_tsne_3d_5kpoints.pickle", 'wb') as f:
      pickle.dump([data_3d], f, pickle.HIGHEST_PROTOCOL)
