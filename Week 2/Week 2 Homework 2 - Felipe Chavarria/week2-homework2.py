#week2-homework2.py

#==============================================================================
#Practice 1
#==============================================================================

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
print("Intercept: {:.3f}".format(intercept))


#in the form y = m*x + b this will be 'm'
coeffs = w[1]
print("Coefficient: {:.3f}".format(coeffs))

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

#==============================================================================
#Practice 2
#==============================================================================

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

#==============================================================================
#Practice 3
#==============================================================================

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#load pickle data
with open("vehicle_price_dataset.pickle", "rb") as f:
    train_data, train_labels, test_data, test_labels = pickle.load(f)

print("Train labels values: \n{}".format(train_labels))
print("Test labels values: \n{}".format(test_labels))

print("Length of train_labels:\t", len(train_labels))
print("Length of test_labels:\t", len(test_labels))

# Show a histograms comparing labels train and test labels from price and odometer
train_data.hist(column=["year", "odometer"], bins=50)
plt.suptitle('Train Labels', fontsize=16)
test_data.hist(column=["year", "odometer"], bins=50)
plt.suptitle('Test Labels', fontsize=16)
plt.show()

#==============================================================================
#Practice 4
#==============================================================================

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#load pickle data
with open("vehicle_price_dataset.pickle", "rb") as f:
    train_data, train_labels, test_data, test_labels = pickle.load(f)

#print("Train labels values: \n{}".format(train_labels))
#print("Test labels values: \n{}".format(test_labels))

# Insert a column of ones
train_data["ones"] = 1

# Select columns odometer and ones
cols = ["ones", "odometer"]
X = train_data[cols].values
Y = train_labels.values

# Solve the Normal equations: W = (X' * X)^-1 * X' * Y
XT = np.transpose(X)
W = np.linalg.inv(XT @ X) @ XT @ Y
#print(W)

eq_str = "price = {:.3f} + {:.3f} * odometer".format(W[0], W[1])
print(eq_str)

# Predict new labels for test data
test_data["ones"] = 1
Xn = test_data[cols].values
Y_pred = Xn @ W

# Print the first few predictions
for idx in range(10):
    print("Predicted: {:6.0f} Correct: {:6d}"\
          .format(Y_pred[idx], test_labels.values[idx]))

# Compute the root mean squared error
error = Y_pred - test_labels.values
rmse = (error ** 2).mean() ** .5
print("RMSE: {:.2f}".format(rmse))

#==============================================================================
#Practice 5
#==============================================================================

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#load pickle data
with open("vehicle_price_dataset.pickle", "rb") as f:
    train_data, train_labels, test_data, test_labels = pickle.load(f)

#print("Train labels values: \n{}".format(train_labels))
#print("Test labels values: \n{}".format(test_labels))

# Insert a column of ones
train_data["ones"] = 1

# Select all columns 
X = train_data.values
Y = train_labels.values

# Solve the Normal equations: W = (X' * X)^-1 * X' * Y
XT = np.transpose(X)
W = np.linalg.inv(XT @ X) @ XT @ Y

intercept = W[0]
coeffs = W[1:]

# Get the headers of data
cols = train_data.columns

print("intercept: {:.3f}".format(intercept))

# adds each feature (header) with its  value to a list of tuples
features = []
for i in range(len(coeffs)):
    features.append((cols[i], coeffs[i]))

# print the value of each coefficient for each feature (each of the headers)
print("coeffs:")
for i in features:
    print("\t{:28s}: {:.3f}".format(i[0], i[1]))

# Predict new labels for test data
test_data["ones"] = 1
Xn = test_data.values
Y_pred = Xn @ W

# Predict new labels for train data
Y_pred2 = X @ W
#print("print y pred:", Y_pred2)
#print("print Y:", Y)

# Print the first few predictions
print("predictions for test data")
for idx in range(10):
    print("\tPredicted: {:6.0f} Correct: {:6d}"\
          .format(Y_pred[idx], test_labels.values[idx]))

print("predictions for train data")
for idx2 in range(10):
    print("\tPredicted: {:6.0f} Correct: {:6d}" \
          .format(Y_pred2[idx2], train_labels.values[idx2]))

# Compute the root mean squared error
error = Y_pred - test_labels.values
error2 = Y_pred2 - train_labels.values
rmse = (error ** 2).mean() ** .5
rmse2 = (error2 ** 2).mean() ** .5
print("Training RMSE: {:.2f}".format(rmse2))
print("Test RMSE: {:.2f}".format(rmse))

#==============================================================================
#Practice 6
#==============================================================================

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#load pickle data
with open("credit_card_default_dataset.pickle", "rb") as f:
    train_data, train_labels, test_data, test_labels = pickle.load(f)

# Insert a column of ones
train_data["ones"] = 1

# Select columns
X = train_data.values
Y = train_labels.values

# Solve the Normal equations: W = (X' * X)^-1 * X' * Y
XT = np.transpose(X)
W = np.linalg.inv(XT @ X) @ XT @ Y
#print(W)

intercept = W[0]
coeffs = W[1:]

# Get the headers of data
cols = train_data.columns

print("Without scikitlearn:")
print("intercept: {:.3f}".format(intercept))

# adds each feature (header) with its  value to a list of tuples
features = []
for i in range(len(coeffs)):
    features.append((cols[i], coeffs[i]))

# print the value of each coefficient for each feature (each of the headers)
print("coeffs:")
for i in features:
    print("\t{:28s}: {:.3f}".format(i[0], i[1]))

# Predict new labels for test data
test_data["ones"] = 1
Xn = test_data.values
Y_pred = Xn @ W

# Print the first few predictions
print("Predictions:")
for idx in range(20):
    print("\tPredicted: {:.3f}\t Correct: {:6d}"\
          .format(Y_pred[idx], test_labels.values[idx]))
      
#With scikitlearn
print("Using scikitlearn:")

import numpy as np
import pandas as pd
import pickle
import sklearn.linear_model
import sklearn.metrics

# Create and train a new LinearRegression model
model = sklearn.linear_model.LinearRegression()

# Train it with the training data and labels
model.fit(train_data[cols], train_labels)

# Print the coefficients
print("intercept: {:.3f}".format(model.intercept_))

# adds each feature (header) with its  value to a list of tuples
features2 = []
for i in range(len(model.coef_)):
    features2.append((cols[i], model.coef_[i]))

# print the value of each coefficient for each feature (each of the headers)
print("coeffs:")
for i in features2:
    print("\t{:28s}: {:.3f}".format(i[0], i[1]))

# Predict new labels for test data
Y_pred_proba = model.predict(test_data[cols])

print("Predictions:")
for idx in range(20):
    print("\tPredicted: {:.3f}\t Correct: {:6d}"\
          .format(Y_pred_proba[idx], test_labels.values[idx]))