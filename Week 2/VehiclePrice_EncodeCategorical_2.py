# Introduction to Artificial Intelligence
# Program to encode categorical features, part 2
# By Juan Carlos Rojas

import numpy as np
import pandas as pd

# Load the dataset
df = pd.read_csv("craigslistVehicles_2000_2018.csv", header=0)

# Encode drive
# Will create two dummy variables: 4wd and fwd

mask = df["drive"] == "4wd"
df["drive_4wd"] = np.where(mask, 1, 0)

mask = df["drive"] == "fwd"
df["drive_fwd"] = np.where(mask, 1, 0)

# Plot count
import matplotlib.pyplot as plt
pd.value_counts(df["drive_4wd"]).plot.barh()
plt.title("drive_4wd")

plt.figure()
pd.value_counts(df["drive_fwd"]).plot.barh()
plt.title("drive_fwd")

plt.show()
