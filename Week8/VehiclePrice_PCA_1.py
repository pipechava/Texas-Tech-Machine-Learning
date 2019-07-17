# Introduction to Artificial Intelligence
# PCA Dimensionality Reduction of Vehicle Price dataset, part 1
# By Juan Carlos Rojas

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn.decomposition

# Load the dataset
df = pd.read_csv("craigslistVehicles_2000_2018.csv", header=0)

# Encode all categorical variables
df = pd.get_dummies(df, prefix_sep="_", drop_first=False)
cols = df.columns

# Standardize the data
data_ndarray = df[cols].values.astype("float64")
scaler = sklearn.preprocessing.StandardScaler()
data_scaled_ndarray = scaler.fit_transform(data_ndarray)
data_scaled = pd.DataFrame(data_scaled_ndarray, columns = df[cols].columns)

print("Computing PCA reduction")
pca = sklearn.decomposition.PCA(n_components=2)
data_2d = pca.fit_transform(data_scaled)

# Pick a random subset
subset_len = 30000
total_len = df.shape[0]
reorder_map = np.random.permutation(total_len)
subset_map = reorder_map[0:subset_len]
subset_data = data_2d[subset_map]

# Plot a scatterplot of the two transformed components
plt.plot(subset_data[:,0], subset_data[:,1], ".")
plt.show()

