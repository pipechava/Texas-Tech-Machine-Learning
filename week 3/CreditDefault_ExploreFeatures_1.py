# Introduction to Artificial Intelligence
# Program to explore features in credit default database, part 1
# By Juan Carlos Rojas

# Read the training data from the Pickle file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("credit_card_default.csv", header=0)

# Print the dataframe info
print(df.info())

# Go through all the columns
for col in df.columns:
    if df[col].dtype == "object":
        # If categorical, do a count bar plot
        lm = pd.value_counts(df[col]).plot.bar()
        lm.set_xticklabels(lm.get_xticklabels(),rotation=0)
        plt.title(col)
        plt.show()

    else:
        # If numerical, do a histogram
        df.hist(column=col, bins=100)
        plt.title(col)
        plt.show()
