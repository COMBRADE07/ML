''''

    linear regression with Multiple variable
    equation:
                y =  m1x1 +m2x2 + m3x3 +..... + c

                where,
                        m1 is the 1st slope
                        c is intercept : this point at which line intercept y-axis
                        x1 is features of data. here in this example its 'median_income'


    *** from sklearn.linear_model import LinearRegression()

'''


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# read data
df = pd.read_csv('housing_data.csv')

print(df.columns)
x = df.iloc[:,2:6].values
y = df.iloc[:,-1].values


