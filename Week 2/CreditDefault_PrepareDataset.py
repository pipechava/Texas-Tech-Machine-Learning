# Introduction to Artificial Intelligence
# # Program to prepare the credit default dataset
# By Juan Carlos Rojas

# Read the training data from the Pickle file
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import sklearn.preprocessing

# Load the dataset
df = pd.read_csv("credit_card_default.csv", header=0)

# Encode all categorical variables
df = pd.get_dummies(df, prefix_sep="_", drop_first=True)

print("Before splitting:")
print(df.info())

# Show some histograms
hist_num_bins = 100
df.hist(column=["Age", "Current_Bill"], bins=hist_num_bins)
plt.show()

# Shuffle the data and divide into training and test sets
total_len = len(df)
train_len = int(total_len*0.8)
test_len = total_len - train_len

reorder_map = np.random.permutation(total_len)
train_idx_map = reorder_map[0:train_len]
test_idx_map = reorder_map[train_len:]

train_set = df.iloc[train_idx_map]
test_set = df.iloc[test_idx_map]

# Separate labels from the rest of the input features
train_labels = train_set["Default"]
train_data = train_set.drop(columns="Default")

test_labels = test_set["Default"]
test_data = test_set.drop(columns="Default")

print("Train data:")
print(train_data.info())
print("Test data:")
print(test_data.info())

# Show a histogram of a few variables in the split sets
train_data.hist(column=["Age", "Current_Bill"], bins=hist_num_bins)
test_data.hist(column=["Age", "Current_Bill"], bins=hist_num_bins)
plt.show()

# Extract the ndarrays from the dataframes
train_data_ndarray = train_data.values.astype("float64")
test_data_ndarray = test_data.values.astype("float64")

# Train a standard scaler from the training data
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(train_data_ndarray)

# Scale both the training and test data
train_data_scaled_ndarray = scaler.transform(train_data_ndarray)
test_data_scaled_ndarray = scaler.transform(test_data_ndarray)

# Re-create Pandas dataframes with the scaled data
train_data_scaled = pd.DataFrame(train_data_scaled_ndarray, columns = train_data.columns)
test_data_scaled = pd.DataFrame(test_data_scaled_ndarray, columns = test_data.columns)

# Show a histogram of a few variables in the split sets
train_data_scaled.hist(column=["Age", "Current_Bill"], bins=hist_num_bins)
test_data_scaled.hist(column=["Age", "Current_Bill"], bins=hist_num_bins)
plt.show()

# Archive the training and test datasets into a pickle file
with open("credit_card_default_dataset.pickle", 'wb') as f:
      pickle.dump([train_data_scaled, train_labels, test_data_scaled, test_labels], \
                  f, pickle.HIGHEST_PROTOCOL)
