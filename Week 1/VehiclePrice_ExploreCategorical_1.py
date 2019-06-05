# Introduction to Artificial Intelligence
# Program to explore categorical columns, part 1
# By Juan Carlos Rojas

# Read the training data from the Pickle file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("craigslistVehicles_2000_2018.csv", header=0)

# Plot bar chart of counts in categorical columns
for col in df.select_dtypes(include='object').columns:
    pd.value_counts(df[col]).plot.barh()
    plt.title(col)
    plt.show()
