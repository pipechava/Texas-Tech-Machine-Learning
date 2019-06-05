#Practice 3

import numpy as np

#creats a simple list with the following values
l = [1 ,2, 3, 3, 1, 2, 2, 3, 1]

#transforms the list to an array to use with numpy
x = np.array(l)
print(x)
#reshape the array x to create a 3x3 matrix
x = x.reshape(3, 3)
print(x)
#computes the inverse of matrix x
y = np.linalg.inv(x)
print(y)
#xy = y @ x ---> doesn't work, don't know why.

#multiply matrices x and y to create the identity matrix
xy = np.matmul(y, x)
print(xy)
