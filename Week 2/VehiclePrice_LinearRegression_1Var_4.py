# Introduction to Artificial Intelligence
# Linear regression model of 1 variable, part 4
# By Juan Carlos Rojas

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("craigslistVehicles_2000_2018.csv")

# Encode all categorical variables
df = pd.get_dummies(df, prefix_sep="_", drop_first=True)

# Insert a column of ones to serve as x0
df["ones"] = 1

#
# Compute Price vs. Transmission linear model
#

X = df[["ones", "transmission_manual"]].values
Y = df["price"].values

# Solve the Normal equations: W = (X' * X)^-1 * X' * Y
XT = np.transpose(X)
W = np.linalg.inv(XT @ X) @ XT @ Y

intercept = W[0]
coeffs = W[1:]
eq_str = "price = {:.0f} + {:.1f} * transmission_manual".format(intercept, coeffs[0])
print(eq_str)

# Plot the scatterplot of price vs. transmission, with the trendline on top

minval = min(df["transmission_manual"])
maxval = max(df["transmission_manual"])
x_range = range(minval, maxval+1)
y_pred = [x*coeffs[0]+intercept for x in x_range]

plt.plot(df["transmission_manual"], df["price"], "b.", x_range, y_pred, ":r")
plt.xlabel("transmission_manual")
plt.ylabel("price")
plt.title(eq_str)
plt.show()

