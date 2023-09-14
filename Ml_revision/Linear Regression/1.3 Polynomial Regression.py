'''
    Polynomial Regression:
            degree                  function                alternate names
    1]  0 degree Polynomial   |   y = constant         | constant
    2]  1 degree Polynomial   |   y = mx^1+c           | linear regression is 1 degree polynomial regression
    3]  2 degree Polynomial   |   y = mx^1+mx^2+c      |

    general equation: y = a0 + a1x + a2x^2 +a3x^3 + ...+ anx^n

'''

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


# import data
df = pd.read_csv('Salary_dataset.csv')

# data preprocessing
print(df.head())
print(df.columns)
print(df.isna().sum())
print(df.describe())

# dropping 1st un named column
df.drop(df.columns[0],axis=1,inplace=True)
print(df)


# splitting x and y
x = [df.iloc[:,0].values]
y = df.iloc[:,-1].values
# plt.scatter(x,y)
# # sns.lmplot(x=x,y=y,data=df)
# plt.title('Experience wise salary')
# plt.xlabel('Salary in lpa')
# plt.ylabel('Year of Experience in yr')
# plt.show()


transform = PolynomialFeatures(degree=2)
x = transform.fit_transform(x)

# splitting testing and training data set
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=10)

# train the model
model = LinearRegression()
model.fit(x_train,y_train)

# test model
pred = model.predict(y_test)
print(pred)



