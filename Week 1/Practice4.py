#Practice 4

import pandas as pd

#creates data frame and reads data from file PeriodicTable.csv
df = pd.read_csv("PeriodicTable.csv")

#creates a new collumn in the data frame called 'mass-num ratio' and adds in it the division from 'atomic mass'/'atomic number'
df['mass-num ratio'] = df['atomic mass']/df['atomic number']

#print the first 6 rows from the data frame
print(df.head(6))
