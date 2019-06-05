#Practice 6

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
