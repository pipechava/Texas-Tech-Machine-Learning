# Introduction to Artificial Intelligence
# Linear regression model of 1 variable, part 2
# By Juan Carlos Rojas

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("craigslistVehicles.csv")

# Insert a column of ones to serve as x0
df["ones"] = 1

# Compute Price vs. Year linear model
X = df[["ones", "year"]].values
Y = df["price"].values

# Solve the Normal equations: W = (X' * X)^-1 * X' * Y
XT = np.transpose(X)
W = np.linalg.inv(XT @ X) @ XT @ Y
intercept = W[0]
coeffs = W[1:]

eq_str = "price = {:.0f} + {:.1f} * year".format(intercept, coeffs[0])
print(eq_str)

# Plot the scatterplot of price vs. year, with the trendline on top

minval = min(df["year"])
maxval = max(df["year"])
x_range = range(minval, maxval)
y_pred = [x*coeffs[0]+intercept for x in x_range]

plt.plot(df["year"], df["price"], "b.", x_range, y_pred, ":r")
plt.xlabel("year")
plt.ylabel("price")
plt.title(eq_str)
plt.show()
