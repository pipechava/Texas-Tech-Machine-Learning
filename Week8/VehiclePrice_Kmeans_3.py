# Introduction to Artificial Intelligence
# K-Means Clustering of Vehicle Price dataset, part 3
# By Juan Carlos Rojas

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn.cluster

# Load the dataset
df = pd.read_csv("craigslistVehicles_2000_2018.csv", header=0)
df_original = df

# Encode all categorical variables
df = pd.get_dummies(df, prefix_sep="_", drop_first=False)
cols = df.columns

# Standardize the data
data_ndarray = df[cols].values.astype("float64")
scaler = sklearn.preprocessing.StandardScaler()
data_scaled_ndarray = scaler.fit_transform(data_ndarray)
data_scaled = pd.DataFrame(data_scaled_ndarray, columns = df[cols].columns)

# Do K-Means clustering
n_clusters = 50
print("Computing K-Means with {} clusters".format(n_clusters))
kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters)  
kmeans.fit(data_scaled)
labels = kmeans.labels_

# Pick a random subset of the data
subset_len = 10000
total_len = data_scaled.shape[0]
reorder_map = np.random.permutation(total_len)
subset_map = reorder_map[0:subset_len]

# Compute the silhouette scores for each sample
silhouette_values = sklearn.metrics.silhouette_samples(data_scaled.iloc[subset_map], labels[subset_map])

# For each cluster, compute the mean of its silhouette scores
silhouette_avg_per_cluster = []
for idx in range(n_clusters):

    # Create a subset of the data limited to this cluster
    silhouette_values_this_cluster = silhouette_values[labels[subset_map]==idx]

    # Print the average score for this cluster
    #print("Cluster {}: Avg. Silhouette score: {:.3f}".format(idx, silhouette_values_this_cluster.mean()))
    silhouette_avg_per_cluster.append(silhouette_values_this_cluster.mean())
print("Avg. silhouette_score: {:.3f}".format(silhouette_values.mean()))

# Sort the clusters by silhouette score
cluster_score_plus_score = [(silhouette_avg_per_cluster[i],i) for i in range(n_clusters)]
cluster_score_plus_score.sort(reverse=True)
print("Top clusters:")
for idx in range(10):
    print("Cluster {}  Score: {:.3f}".format(cluster_score_plus_score[idx][1], cluster_score_plus_score[idx][0]))

    # Get a subset of the original data, limited to this cluster
    data_this_cluster = df_original[labels==idx]
    print("  Number of samples: ", data_this_cluster.shape[0])

    for col in data_this_cluster.columns:
        if data_this_cluster[col].dtype == "object":
            # If categorical, do a count bar plot
            plt.figure()
            lm = pd.value_counts(data_this_cluster[col]).plot.bar()
            lm.set_xticklabels(lm.get_xticklabels(),rotation=0)
            plt.title(col)
            #plt.show()

        else:
            # If numerical, do a histogram
            data_this_cluster.hist(column=col, bins=100)
            plt.title(col)
            #plt.show()
    plt.show()

