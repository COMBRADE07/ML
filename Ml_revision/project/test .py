import pandas as pd

from sklearn.preprocessing import OneHotEncoder

onehot = OneHotEncoder()

df = pd.read_csv('cleaned.csv')
print(df.head())
print(df.Locality)

dummies = pd.get_dummies(df.Locality)
print(dummies.head())