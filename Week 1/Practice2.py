#Practice 2

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
