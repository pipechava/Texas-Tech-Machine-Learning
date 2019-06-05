# Introduction to Artificial Intelligence
# Program to explore categorical columns, part 2
# By Juan Carlos Rojas

# Read the training data from the Pickle file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("craigslistVehicles.csv", header=0)

# Plot box plots of price vs. category
for col in df.select_dtypes(include='object').columns:
    sns.boxplot(data=df, y=col, x="price", fliersize=1)
    plt.title("Price vs. {}".format(col))
    plt.show()
