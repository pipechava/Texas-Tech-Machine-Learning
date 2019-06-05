# Introduction to Artificial Intelligence
# Program to prepare the vehicle price dataset
# By Juan Carlos Rojas

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("craigslistVehicles_2000_2018.csv", header=0)

# Encode all categorical variables
df = pd.get_dummies(df, prefix_sep="_", drop_first=True)

#print("Before splitting:")
print(df.info())

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
train_labels = train_set["price"]
train_data = train_set.drop(columns="price")

test_labels = test_set["price"]
test_data = test_set.drop(columns="price")

#print("Train data:")
#print(train_data.info())
#print("Test data:")
#print(test_data.info())


## Show a histogram of a few variables in the split sets
#train_data.hist(column=["year", "odometer"], bins=50)
#test_data.hist(column=["year", "odometer"], bins=50)
#plt.show()

#print("Train data:")
#print(train_data)
#print("train_labels:")
#print(train_labels)
#print("Test data:")
#print(test_data)
#print("test_labels:")
#print(test_labels)

## Archive the training and test datasets into a pickle file
#import pickle
#with open("vehicle_price_dataset.pickle", 'wb') as f:
#      pickle.dump([train_data, train_labels, test_data, test_labels], \
#                  f, pickle.HIGHEST_PROTOCOL)
