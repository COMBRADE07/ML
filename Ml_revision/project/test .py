import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# onehot = OneHotEncoder(sparse_output=False)
#
df = pd.read_csv('cleaned.csv')
# print(df.head())
#
# df2 = onehot.fit_transform(df.Locality.values.reshape(-1, 1))
# print(df2)


# transform
onehot_transformer = OneHotEncoder(sparse_output=False,drop='first')
transform1_features = ['Locality','Furnishing']
transformer = ColumnTransformer([
    ('transform1',onehot_transformer,transform1_features)
],remainder='passthrough')

print(len(df.Locality.unique()))
print(len(df.Furnishing.unique()))
df = transformer.fit_transform(df)
print(df)