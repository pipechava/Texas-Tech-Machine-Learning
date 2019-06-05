#Practice 5

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
