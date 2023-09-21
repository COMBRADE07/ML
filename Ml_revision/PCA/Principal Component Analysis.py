'''
    Principal Component Analysis:

        - It is technique to reduce dimension
        - It is a process of figuring out most important feature or Principal component
          that has most impact on target variables.
'''

import pandas as pd
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
dataset = load_digits()

df = pd.DataFrame(dataset.data,columns=dataset.feature_names)

# df['target'] = dataset.target
x = df.iloc[:,:].values
y = dataset.target

# scaling is very important while performing PCA
scale = MinMaxScaler()
x = scale.fit_transform(x)

# splitting training and testing
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

model = LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)
print(model.score(x_train,y_train))