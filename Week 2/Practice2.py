#Practice 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# dataframe from craigslistVehicles_2000_2018.csv
df = pd.read_csv("craigslistVehicles_2000_2018.csv")

# Encode all categorical variables, this sets columns in the dataframe with each value in each column minus one
df = pd.get_dummies(df, prefix_sep="_", drop_first=True)

# Insert a column of ones to serve as x0
df["ones"] = 1

# Compute Price vs manufacturer_audi

#get column values and stores it in X Y for each column
X = df[["ones", "manufacturer_audi"]].values
Y = df["price"].values

# Solve the Normal equations: W = (X' * X)^-1 * X' * Y
XT = np.transpose(X)
W = np.linalg.inv(XT @ X) @ XT @ Y

# gets intercept value = value of Y when X = 0, in Y = M*X+B this will be B
intercept = W[0]

# gets coefficients values = in Y = M*X+B this will be M
coeffs = W[1:]

eq_str = "price = {:.0f} + {:.1f} * manufacturer_audi".format(intercept, coeffs[0])
print(eq_str)

# Plot the scatterplot of price vs. manufacturer_audi, with the trendline on top

minval = min(df["manufacturer_audi"])
maxval = max(df["manufacturer_audi"])

# gets a list of x axis
x_range = range(minval, maxval+1)

# computes all y values for all x values
y_pred = [x*coeffs[0]+intercept for x in x_range]

# plots Price vs manufacturer_audi
plt.plot(df["manufacturer_audi"], df["price"], "b.", x_range, y_pred, ":r")

# labels x, y and title and show plot
plt.xlabel("manufacturer_audi")
plt.ylabel("price")
plt.title(eq_str)
plt.show()