#Practice3

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#load pickle data
with open("vehicle_price_dataset.pickle", "rb") as f:
    train_data, train_labels, test_data, test_labels = pickle.load(f)


print("Train labels values: \n{}".format(train_labels))
print("Test labels values: \n{}".format(test_labels))

print("Length of train_labels:\t", len(train_labels))
print("Length of test_labels:\t", len(test_labels))

# Show a histograms comparing labels train and test labels from price and odometer
train_data.hist(column=["year", "odometer"], bins=50)
plt.suptitle('Train Labels', fontsize=16)
test_data.hist(column=["year", "odometer"], bins=50)
plt.suptitle('Test Labels', fontsize=16)
plt.show()