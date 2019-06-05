# Introduction to Artificial Intelligence
# Linear regression model of 1 variable, part 1
# By Juan Carlos Rojas

import numpy as np
import pandas as pd

df = pd.read_csv("craigslistVehicles_2000_2018.csv")

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

print("Linear Model: price = {:.0f} + {:.1f} * year"\
      .format(intercept, coeffs[0]))

