#Practice 1

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
    print(j)
