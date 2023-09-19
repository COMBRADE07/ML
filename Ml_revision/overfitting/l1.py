# Lasso regularization | l1 regularization

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
iris = load_iris()

df = pd.DataFrame(iris.data,columns=iris.feature_names)
df['target'] = iris.target
print(df.head())

x = df.iloc[:,0:4].values
y = df.iloc[:,-1].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
model = LinearRegression()
model.fit(x_train,y_train)

lasso_model = Lasso(alpha=2.0,max_iter=120,tol=0.1)
lasso_model.fit(x_train,y_train)
print(lasso_model.score(x_train,y_train))
print(lasso_model.score(x_test,y_test))