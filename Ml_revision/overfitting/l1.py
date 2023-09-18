# Lasso regularization | l1 regularization

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# reading data
data = pd.read_csv('housing_data.csv')

# data overview
print(data.head())
print(data.columns)
print(data.describe())


# extracting dependent and independent variables
x = data[['median_income']].values
y = data[['median_house_value']].values


# splitting dataset
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)

print("length of x_train: ",len(x_train))
print("length of x_test: ",len(x_test))

# model building and training
model = LinearRegression()
model.fit(x_train,y_train)

# test the model
pred = model.predict(y_test)

print(pred)

# checking accuracy
s = model.score(x_test,y_test)
print(s*100)