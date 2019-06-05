# Introduction to Artificial Intelligence
# Program to explore features in credit default database, part 2
# By Juan Carlos Rojas

# Read the training data from the Pickle file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("credit_card_default.csv", header=0)

# Do scatterplot with colors depending on labels
xcol = "Credit_Limit"
ycol = "Current_Bill"

plt.scatter(df[xcol], df[ycol], c=df["Default"], alpha=0.3)
plt.xlabel(xcol)
plt.ylabel(ycol)
plt.show()


