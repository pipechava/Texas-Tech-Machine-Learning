# Introduction to Artificial Intelligence
# K-Means Clustering of Credit Default detaset, part 2b
# By Juan Carlos Rojas

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn.cluster

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

avg_score_list=[]

min_k = 5
max_k = 100
k_step = 5

for n_clusters in range(min_k, max_k, k_step):

    # Do K-Means clustering
    print("Computing K-Means with {} clusters".format(n_clusters))
    kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters)  
    kmeans.fit(data_scaled)
    labels = kmeans.labels_

    # Compute the average silhouette score
    #print("Computing silhouette scores")

    # Pick a random subset of the data
    subset_len = 30000

    total_len = data_scaled.shape[0]
    reorder_map = np.random.permutation(total_len)
    subset_map = reorder_map[0:subset_len]

    # Compute the silhouette scores for each sample
    silhouette_values = sklearn.metrics.silhouette_samples(data_scaled.iloc[subset_map], labels[subset_map])

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

