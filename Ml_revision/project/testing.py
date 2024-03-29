# -*- coding: utf-8 -*-
"""Delhi_House_Price_Predictor.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1o3JLfS0iStAIYRAIXWTfEMFhfsxy_Bxn

# testing
"""

import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from myfunction import Extracttion, Encoder,Outlier
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge

# load dataset
df = pd.read_csv('MagicBricks.csv')

df.head()

# Understand the data
df.columns

df.describe()

df.info()

# explore furnishing column
df.groupby('Furnishing')['Status'].agg('count')

# step 2. drop unnecessary columns
df = df.drop(['Parking', 'Status', 'Transaction', 'Type', 'Per_Sqft'], axis=1)

# Checking for Null values
df.isna().sum()

# step 3.fill them with mean
mean_bathroom = df.Bathroom.mean()
# filling bathroom value with mean
df['Bathroom'] = df['Bathroom'].fillna(mean_bathroom)
# print(df.Furnishing.value_counts())
df['Furnishing'] = df.Furnishing.fillna(method='bfill')
df.isna().sum()

# conclusion
'''
    - Locality Columns
    - In this dataset thier is major** problem with locality column some of thier value are fine but,
    - when you observe close then you find some of fields contains lot of unnecesary information
    - "here we want to clean locality columns and get exact location from it."
    - lets do it.
'''

# extracting propper locations
ext = Extracttion()

df['Locality'] = df['Locality'].apply(ext.extract_loc)
df['Locality'] = df['Locality'].apply(ext.remove_nums)
df.Locality.head(10)

df.Locality.value_counts()

df1 = df.copy()

# replace locality count whoes count is less than 5.
loc_summarry = df1.groupby('Locality')['Locality'].agg('count').sort_values(ascending=False)
lessthan = loc_summarry[loc_summarry <= 5]
ll = df1.Locality.value_counts()

# update dataframe which has less than 5 count
df1.Locality = df1.Locality.apply(lambda x: 'other' if x in lessthan else x)

df.Locality.value_counts()

# checking all operation is fine or not
df1.head()

df1.isna().sum()

print("max_bathroom: ",df1.Bathroom.max())
print("max_Price: ",df1.Price.max())
print("min_Price: ",df1.Price.min())
print("min_Area: ",df1.Area.min())
print("max_Area: ",df1.Area.max())

# extarcting minimum price row
df.loc[df['Price'].idxmin()]

# extarcting max price row
df.loc[df['Price'].idxmax()]

# extarcting max bathroom row
df.loc[df['Bathroom'].idxmax()]

# extract categorical columns for encoding
dfc = df1.copy()
dfc=dfc.drop(['Area','Price','Bathroom','BHK'],axis=1)
dfc.head()

# extract numerical columns
dfn = df1.copy()
dfn=dfn.drop(['Furnishing','Locality',],axis=1)
dfn.head()

# encoding
encoder = Encoder()
encodeddf = encoder.onehot_encoder(dfc,'Furnishing','Locality')
encodeddf.head()

# testng cells
# dfn.isna().sum() # test done
# dfc.isna().sum() # test done
# encodeddf.isna().sum() # test done

# outlier detection
otl = Outlier()
otl.check_outliers(dfn,['Area','BHK','Bathroom','Price'])

# remove outliers
# otl.remove_outliers(df3)
def remove_outliers(df):
    df1 = df.select_dtypes(include=['number'])  # for numbers
    df2 = df.select_dtypes(exclude=['number'])  # categoricalcolumns
    #  here we used IQR method
    ll = 0.25
    ul = 0.75

    Q1 = df1.quantile(q=ll)
    Q3 = df1.quantile(q=ul)
    IQR = Q3 - Q1

    lb = Q1 - 1.5 * IQR
    ub = Q3 + 1.5 * IQR
    df1 = df1[(df1 >= lb) & (df1 <= ub)]
    df = pd.concat([df1, df2], axis=1)
    return df

dff=remove_outliers(dfn)

dff.head()

# outlier checking
otl = Outlier()
otl.check_outliers(dff,['Area','BHK','Bathroom','Price'])

dff.isna().sum()

# filling bathroom value with mean
dff['Bathroom'] = dff['Bathroom'].fillna(dff.Bathroom.mean())
dff['Area'] = dff['Area'].fillna(method='ffill')
dff['BHK'] = dff['BHK'].fillna(dff.Price.mean())
dff['Price'] = dff['Price'].fillna(dff.Price.mean())
# print(df.Furnishing.value_counts())
dff.isna().sum()

df3 = pd.concat([dff,encodeddf],axis=1)

df3.head()

df3.isna().sum()

# extarct x and y
x = df3.drop(['Price'],axis=1)
y = df3['Price']

# Define your feature matrix 'x' and target variable 'y' here

model_params = {
    'linear_regression': {
        'model': LinearRegression(),
        'params': {}
    },
    'svr': {
        'model': SVR(),
        'params': {
            'C': [1, 10, 20],
            'kernel': ['linear', 'poly', 'rbf']
        }
    },
    'random_forest_regressor': {
        'model': RandomForestRegressor(),
        'params': {
            'n_estimators': [10, 50, 100],
            'max_depth': [None, 10, 20, 30]
        }
    }
}

score = []
for mn, mp in model_params.items():
    gsv = GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    gsv.fit(x, y)  # Make sure 'x' and 'y' are defined
    score.append({
        'model': mn,
        'best_score': gsv.best_score_,
        'best_params': gsv.best_params_
    })

sc = pd.DataFrame(data=score, columns=['model', 'best_score', 'best_params'])
print(sc.head())


# # splitting dataset into training and testing
# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)
#
# scaler = StandardScaler()
# x_train_scaled = scaler.fit_transform(x_train)
# x_test_scaled = scaler.transform(x_test)
#
# print('Training set lenght:', len(x_train),'\nTesting set lenght:', len(x_test))
# print(x_train)
# lr= LinearRegression()
# lr.fit(x_train,y_train)

# print(lr.score(x_test,y_test))
