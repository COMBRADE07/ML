import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# onehot = OneHotEncoder(sparse_output=False)
#
df = pd.read_csv('cleaned.csv')
# print(df.head())
#
# df2 = onehot.fit_transform(df.Locality.values.reshape(-1, 1))
# print(df2)


# transform
'''
    --
        Transformer class use to apply different type of transformation on dataframe
        using this we can reduce multiple line of code.
        also it return 2d array
    --

for transformer we have to pass two paramerter
1. transformer:{(transformer_name,transformer,columns),}::
    - transformer_name: e.g. 'tf1'
    - transformer:e.g OneHotEncoder()
    - columns: here we have to pass dataframe columns
2. remainder{'passthrough','drop'}
    - 'passthrough': this will keep all remaining columns as it.
    - 'drop': this will drop remaining columns
    
'''
onehot_transformer = OneHotEncoder(sparse_output=False,drop='first')
transform1_features = ['Locality','Furnishing']
transformer = ColumnTransformer([
    ('transform1',onehot_transformer,transform1_features),
    # ('transform2',SimpleImputer(strategy='median'),['BHK'])
],remainder='passthrough')

print(len(df.Locality.unique()))
print(len(df.Furnishing.unique()))
df = transformer.fit_transform(df)
print(df)