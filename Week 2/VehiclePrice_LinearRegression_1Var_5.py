# Introduction to Artificial Intelligence
# Linear regression model of 1 variable, part 5
# By Juan Carlos Rojas

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("craigslistVehicles_2000_2018.csv")

# Encode all categorical variables
df = pd.get_dummies(df, prefix_sep="_", drop_first=True)

# Insert a column of ones to serve as x0
df["ones"] = 1

# Loop through all the dummy variables
# They can be recognized for using uint8 types

for col in df.select_dtypes(include='uint8').columns:

    X = df[["ones", col]].values
    Y = df["price"].values

    # Solve the Normal equations: W = (X' * X)^-1 * X' * Y
    XT = np.transpose(X)
    W = np.linalg.inv(XT @ X) @ XT @ Y

    intercept = W[0]
    coeffs = W[1:]
    eq_str = "price = {:.0f} + {:.2f} * {}".format(intercept, coeffs[0], col)
    print(eq_str)

    # Plot a boxplot of price vs. this column
    #plt.figure()
    lm = sns.boxplot(data=df, x=col, y="price", fliersize=1)

    # Get plot axes and compute x offset
    plot_xaxes = lm.axes.get_xlim()
    plot_minx = plot_xaxes[0]
    plot_maxx = plot_xaxes[1]

    # Compute scaling factor between plot plot scale, and model scale
    minval = min(df[col])
    maxval = max(df[col])
    offset = minval - plot_minx
    scale_factor = (maxval - minval) / (plot_maxx - plot_minx)

    # Compute the trendline
    x_range_plot = range(int(plot_minx), int(plot_maxx)+1)
    x_range_model = [x*scale_factor+offset for x in x_range_plot]
    y_pred = [x*coeffs[0]+intercept for x in x_range_model]

    # Plot trendline on top
    #plt.plot(x_range_plot, y_pred, ":r", linewidth=3.3)
    #plt.title(eq_str)
    #plt.show()


