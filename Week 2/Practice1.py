#Practice 1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#reads "craigslistVehicles.csv" file
df = pd.read_csv("craigslistVehicles.csv")

#add column "ones" to data frame with ones in it
df["ones"] = 1

#creates a matrix "x" with columns ones and odometer
x = df[["ones", "odometer"]].values
#print("print x matrix")
#print(x)

#creates a vector "y" which has prices
y = df["price"].values
#print("print y vector")
#print(y)

#transpose matrx x
xt = np.transpose(x)
#print("print xt matrix")
#print(xt)

#solves normal equation: W = (X' * X)^-1 * X' * Y
w = np.matmul(np.matmul(np.linalg.inv(np.matmul(xt, x)),xt),y)
#print("print w")
#print(w)

#in the form y = m*x + b this will be 'b'
intercept = w[0]
#print("print intercept")
#print(intercept)

#in the form y = m*x + b this will be 'm'
coeffs = w[1]
#print("print coeffs")

print("Linear Model: price = {:.0f} + {:.3f} * odometer".format(intercept, coeffs))

minval = min(df["odometer"])
maxval = max(df["odometer"])

# gets a list of x axis
x_range = range(minval, maxval)

# computes all y values for all x values
y_pred = [x*coeffs+intercept for x in x_range]

# plot odometer vs price
plt.plot(df["odometer"], df["price"], "b.", x_range, y_pred, ":r")

#labels x, y and title and show plot
plt.xlabel("odometer")
plt.ylabel("price")
plt.title("price = {:.0f} + {:.3f} * odometer".format(intercept, coeffs))
plt.show()

#=================================
#from here code to plot boxplot:
#=================================

#group odometer data in 19 groups
df["odometer_discrete"] = pd.cut(df["odometer"], 19)
#creates the boxplot
lm = sns.boxplot(data=df, x="odometer_discrete", y="price", fliersize=1)
#rotate x labels 30 degrees
lm.set_xticklabels(lm.get_xticklabels(),rotation=30)
#adds title
plt.title("price = {:.0f} + {:.3f} * odometer".format(intercept, coeffs))
#adds legend
lm.legend(['trend', 'boxplot'], facecolor='w')

# Return the x-axis view limits, [0]is the far left axis and [1] is the far right axis
plot_xaxes = lm.axes.get_xlim()

# assignes minx to the most left x axis
plot_minx = plot_xaxes[0]
#print(plot_minx)

# assignes maxx to the most right x axis
plot_maxx = plot_xaxes[1]
#print(plot_maxx)

# Compute scaling factor between plot plot scale, and model scale

# computes the value of the range of each x group in the plot
scale_factor = (maxval - minval) / (plot_maxx - plot_minx)
#print(scale_factor)


offset = minval - plot_minx
#print(offset)

# Compute the trendline

# sets x_range_plot to be a tuple with first and last positions in a list and adds 1 to last position to extend the trendline, otherwise it will end in the bin second to last
x_range_plot = range(int(plot_minx), int(plot_maxx)+1)
#print(x_range_plot)

#gets each number of the range of every bin and stores it in x_range_model
x_range_model = [x*scale_factor+offset for x in x_range_plot]
#print(x_range_model)

# computes y = m*x +b
y_pred = [x*coeffs+intercept for x in x_range_model]

# Plot trendline on top of the boxplot
plt.plot(x_range_plot, y_pred, ":r", linewidth=3.3)

#shows plot
plt.show()