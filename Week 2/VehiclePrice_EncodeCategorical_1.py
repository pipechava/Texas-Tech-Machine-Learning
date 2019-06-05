# Introduction to Artificial Intelligence
# Program to encode categorical features, part 1
# By Juan Carlos Rojas

import numpy as np
import pandas as pd

# Load the dataset
df = pd.read_csv("craigslistVehicles_2000_2018.csv", header=0)

# Encode transmission
# Since this is binary, simply replace "manual" with 0,
# and "automatic" with 1

mask = df["transmission"] == "automatic"
df["transmission_automatic"] = np.where(mask, 1, 0)

# Plot count
import matplotlib.pyplot as plt
pd.value_counts(df["transmission_automatic"]).plot.barh()
plt.title("transmission_automatic")
plt.show()
