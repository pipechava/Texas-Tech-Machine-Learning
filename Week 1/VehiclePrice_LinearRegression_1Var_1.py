# Introduction to Artificial Intelligence
# Linear regression model of 1 variable, part 1
# By Juan Carlos Rojas

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#df = pd.read_csv("craigslistVehicles.csv")

# Insert a column of ones to serve as x0
#df["ones"] = 1

# Compute Price vs. Year linear model
#X = df[["ones", "year"]].values
#Y = df["price"].values

# Solve the Normal equations: W = (X' * X)^-1 * X' * Y
#XT = np.transpose(X)
#W = np.linalg.inv(XT @ X) @ XT @ Y
#intercept = W[0]
#coeffs = W[1:]

#print("Linear Model: price = {:.0f} + {:.1f} * year"\
#      .format(intercept, coeffs[0]))


df = pd.read_csv("craigslistVehicles.csv")

df["ones"] = 1

x = df[["ones", "year"]].values
print("print x matrix")
print(x)

y = df["price"].values
print("print y vector")
print(y)

xt = np.transpose(x)
print("print xt matrix")
print(xt)

w = np.matmul(np.matmul(np.linalg.inv(np.matmul(xt, x)),xt),y)
print("print w")
print(w)

for i in w:
    print(i)

intercept = w[0]
print("print intercept")
print(intercept)

coeffs = w[1]
print("print coeffs")
print(coeffs)

print("Linear Model: price = {:.0f} + {:.1f} * year".format(intercept, coeffs))

minval = min(df["year"])
maxval = max(df["year"])
x_range = range(minval, maxval)
y_pred = [x*coeffs+intercept for x in x_range]

plt.plot(df["year"], df["price"], "b.", x_range, y_pred, ":r")
plt.xlabel("year")
plt.ylabel("price")
plt.title("price = {:.0f} + {:.1f} * year".format(intercept, coeffs))
plt.show()
