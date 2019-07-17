# Introduction to Artificial Intelligence
# t-SNE Dimensionality Reduction of Credit Default detaset, part 1
# By Juan Carlos Rojas

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn.manifold
import time

# Load the dataset
df = pd.read_csv("credit_card_default.csv", header=0)

# Encode all categorical variables
df = pd.get_dummies(df, prefix_sep="_", drop_first=False)
cols = df.columns

# Standardize the data
data_ndarray = df[cols].values.astype("float64")
scaler = sklearn.preprocessing.StandardScaler()
data_scaled_ndarray = scaler.fit_transform(data_ndarray)
data_scaled = pd.DataFrame(data_scaled_ndarray, columns = df[cols].columns)

# Compute the t-SNE transformation into 2 dimensions
# Do it over a random subset
subset_len = 10000
total_len = df.shape[0]
reorder_map = np.random.permutation(total_len)
subset_map = reorder_map[0:subset_len]
subset_data = data_scaled.iloc[subset_map]

print("Computing t-SNE reduction into 2 dimensions")
start_time = time.time()
tsne = sklearn.manifold.TSNE(n_components=2)
data_2d = tsne.fit_transform(subset_data)
end_time = time.time()
print("Completed in: ", end_time - start_time);

# Plot a scatterplot of the two transformed components
plt.plot(data_2d[:,0], data_2d[:,1], ".")
plt.show()



