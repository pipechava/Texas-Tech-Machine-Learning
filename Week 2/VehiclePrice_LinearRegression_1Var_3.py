# Introduction to Artificial Intelligence
# Linear regression model of 1 variable, part 3
# By Juan Carlos Rojas

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("craigslistVehicles_2000_2018.csv")

# Insert a column of ones to serve as x0
df["ones"] = 1

#
# Compute Price vs. Year linear model
#

#"""
X = df[["ones", "year"]].values
Y = df["price"].values

# Solve the Normal equations: W = (X' * X)^-1 * X' * Y
XT = np.transpose(X)
W = np.linalg.inv(XT @ X) @ XT @ Y
intercept = W[0]
coeffs = W[1:]

eq_str = "price = {:.0f} + {:.1f} * year".format(intercept, coeffs[0])
print(eq_str)

# Plot a discretized boxplot
plt.figure()
df["year_discrete"] = pd.cut(df["year"], 19)
lm = sns.boxplot(data=df, x="year_discrete", y="price", fliersize=1)
lm.set_xticklabels(lm.get_xticklabels(),rotation=30)

# Get plot axes and compute x offset
plot_xaxes = lm.axes.get_xlim()
plot_minx = plot_xaxes[0]
plot_maxx = plot_xaxes[1]

# Compute scaling factor between plot plot scale, and model scale
minval = min(df["year"])
maxval = max(df["year"])
offset = minval - plot_minx
scale_factor = (maxval - minval) / (plot_maxx - plot_minx)

# Compute the trendline
x_range_plot = range(int(plot_minx), int(plot_maxx)+1)
x_range_model = [x*scale_factor+offset for x in x_range_plot]
y_pred = [x*coeffs[0]+intercept for x in x_range_model]

# Plot trendline on top
plt.plot(x_range_plot, y_pred, ":r", linewidth=3.3)

plt.title(eq_str)
plt.show()
#"""

#
# Compute Price vs. Odometer linear model
#

#"""
X = df[["ones", "odometer"]].values
Y = df["price"].values

# Solve the Normal equations: W = (X' * X)^-1 * X' * Y
XT = np.transpose(X)
W = np.linalg.inv(XT @ X) @ XT @ Y
intercept = W[0]
coeffs = W[1:]

eq_str = "price = {:.0f} + {:.3f} * odometer".format(intercept, coeffs[0])
print(eq_str)

# Plot a discretized boxplot
plt.figure()
df["odometer_discrete"] = pd.cut(df["odometer"], 19)
lm = sns.boxplot(data=df, x="odometer_discrete", y="price", fliersize=1)
lm.set_xticklabels(lm.get_xticklabels(),rotation=30)

# Get plot axes and compute x offset
plot_xaxes = lm.axes.get_xlim()
plot_minx = plot_xaxes[0]
plot_maxx = plot_xaxes[1]

minval = min(df["odometer"])
maxval = max(df["odometer"])

# Compute scaling factor between plot plot scale, and model scale
scale_factor = (maxval - minval) / (plot_maxx - plot_minx)
offset = minval - plot_minx

# Compute the trendline
x_range_plot = range(int(plot_minx), int(plot_maxx)+1)
x_range_model = [x*scale_factor+offset for x in x_range_plot]
y_pred = [x*coeffs[0]+intercept for x in x_range_model]

# Plot trendline on top
plt.plot(x_range_plot, y_pred, ":r", linewidth=3.3)

plt.title(eq_str)
plt.show()
#"""
