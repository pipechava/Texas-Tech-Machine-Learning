# Introduction to Artificial Intelligence
# Explore numerical variables, part 2
# By Juan Carlos Rojas

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("craigslistVehicles_2000_2018.csv")

# Discretize the year into bins and box-plot
plt.figure()
df["year_discrete"] = pd.cut(df["year"], 19)
sns.boxplot(data=df, y="year_discrete", x="price", fliersize=1)
plt.title("Price vs. Year")

# Discretize the odometer into bins and box-plot
plt.figure()
df["odometer_discrete"] = pd.cut(df["odometer"], 10)
sns.boxplot(data=df, y="odometer_discrete", x="price", fliersize=1)
plt.title("Price vs. odometer")

plt.show()





