# Introduction to Artificial Intelligence
# Program to scale the vehicle price dataset
# By Juan Carlos Rojas

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn.preprocessing

# Load the training and test data from the Pickle file
with open("vehicle_price_dataset.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

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
train_data_scaled.hist(column=["year", "odometer"], bins=50)
test_data_scaled.hist(column=["year", "odometer"], bins=50)
plt.show()

# Archive the training and test datasets into a pickle file
with open("vehicle_price_dataset_scaled.pickle", 'wb') as f:
      pickle.dump([train_data_scaled, train_labels, test_data_scaled, test_labels], \
                  f, pickle.HIGHEST_PROTOCOL)
