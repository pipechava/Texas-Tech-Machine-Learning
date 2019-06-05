#week1-homework1.py

#==============================================================================
#Practice 1
#==============================================================================

#initialize both lists x and y
x = []
y = []

#starts a loop from 0 to 100
for i in range(101):
    #adds all numbers from 0 to 100 to list x
    x.append(i)
    #from the i position grabs that number from list x and does *3+5
    res = x[i]*3+5
    #add result from variable res to list y
    y.append(res)

#print all values from list y
for j in y:
    print(j, end = ', ')

#==============================================================================
#Practice 2
#==============================================================================

import matplotlib.pyplot as plt
from random import randint

#initialize lists
x = []
y = []
yr = []

#loop to add values from -10 to 10
for i in range(-10, 10):
    #adds numbers to list x
    x.append(i)

# goes over all values in list x
for j in x:
    res = 3*j+5
    #adds random integers from -5 to 5
    res2 = res + randint(-5, 5)
    #adds res to y list
    y.append(res)
    #adds res2 to yr list
    yr.append(res2)

#plots the blue line which contains y list values
plt.plot(x, y, '--b')
#plots red dots which contains single values in yr list
plt.plot(x, yr, "r.")
#shows the plot
plt.show()

#==============================================================================
#Practice 3
#==============================================================================

import numpy as np

#creats a simple list with the following values
l = [1 ,2, 3, 3, 1, 2, 2, 3, 1]

#transforms the list to an array to use with numpy
x = np.array(l)
print("Array x:")
print(x)
#reshape the array x to create a 3x3 matrix
x = x.reshape(3, 3)
print("Matrix x:")
print(x)
#computes the inverse of matrix x
y = np.linalg.inv(x)
print("Inverse of matrix x:")
print(y)
#xy = y @ x ---> doesn't work, don't know why.

#multiply matrices x and y to create the identity matrix
xy = np.matmul(y, x)
print("Result of multiplaction of matrices x and y:")
print(xy)

#==============================================================================
#Practice 4
#==============================================================================

import pandas as pd

#creates data frame and reads data from file PeriodicTable.csv
df = pd.read_csv("PeriodicTable.csv")

#creates a new collumn in the data frame called 'mass-num ratio' and adds in it the division from 'atomic mass'/'atomic number'
df['mass-num ratio'] = df['atomic mass']/df['atomic number']

#print the first 6 rows from the data frame
print(df.head(6))

#==============================================================================
#Practice 5
#==============================================================================

import pandas as pd
#import matplotlib as plt ---> this import will not show the histogram, use instead 'import matplotlib.pyplot as plt'
import matplotlib.pyplot as plt

#reads craigslistVehicles.csv file and puts it in 'df' data frame variable
df = pd.read_csv("craigslistVehicles.csv")

#creates a histogram for each column with values: year, odometer, price
df.hist(column=['year'], bins=50)
df.hist(column=['odometer'], bins=50)
df.hist(column=['price'], bins=50)

#shows the 3 histograms
plt.show()

#==============================================================================
#Practice 6
#==============================================================================

import pandas as pd
#import matplotlib as plt ---> this import will not show the histogram, use instead 'import matplotlib.pyplot as plt'
import matplotlib.pyplot as plt

#reads craigslistVehicles.csv file and puts it in 'df' data frame variable
df = pd.read_csv("craigslistVehicles.csv")

#creates a scatter plot with data from columns year and price comparing each column
df.plot.scatter(x='year',y='price',title='Price vs Year')

#creates a scatter plot with data from columns odometer and price comparing each column
df.plot.scatter(x='odometer',y='price',title='Price vs Odometer')

#shows the 2 scatter plots
plt.show()
