# Introduction to Artificial Intelligence
# Program to encode categorical features, part 3
# By Juan Carlos Rojas

import numpy as np
import pandas as pd

# Load the dataset
df = pd.read_csv("craigslistVehicles_2000_2018.csv", header=0)

# Encode all categorical variables
df = pd.get_dummies(df, prefix_sep="_", drop_first=True)

print(df.info())
