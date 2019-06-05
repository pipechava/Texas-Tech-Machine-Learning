# Introduction to Artificial Intelligence
# Program to explore features in credit default database, part 2
# By Juan Carlos Rojas

# Read the training data from the Pickle file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("credit_card_default.csv", header=0)

# Go through all the columns
for col in df.columns:
    if col == "Default":
        continue
    
    if df[col].dtype == "object":
        # If categorical, do a bar plot of counts
        lm = sns.barplot(data=df, x=col, y="Default")
        lm.set_xticklabels(lm.get_xticklabels(),rotation=30)
        plt.title("Default vs. {}".format(col))
        plt.show()

    else:
        # If numerical, do a discretized bar plot
        df[col] = pd.qcut(df[col], 10, duplicates="drop")
        lm = sns.barplot(data=df, x=col, y="Default")
        lm.set_xticklabels(lm.get_xticklabels(),rotation=30)
        plt.title("Default vs. {}".format(col))
        plt.show()
