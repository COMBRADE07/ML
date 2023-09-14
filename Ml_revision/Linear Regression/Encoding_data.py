'''
    Encoding of categorical data
    1] LabaleEncoder()
    2] OneHotEncoder()
    3]pd.getDummies()


'''

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import pandas as pd

# create data frame
data = {'name':['A','B','C','D','E'],
        'age':[10,20,30,2,5]}
df = pd.DataFrame(data=data)

x = df.iloc[:,0]
x1 = df.iloc[:,0]
x2 = df.iloc[:,0]

# 1] LabelEncoder()
lable = LabelEncoder()
x = lable.fit_transform(x)
print(x)

# 2] OneHotEncoder()
onehot = OneHotEncoder()
x1 = onehot.fit_transform(x1.values.reshape(-1,1)).toarray()
print(x1)