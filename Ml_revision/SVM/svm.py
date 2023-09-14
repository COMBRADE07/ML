import pandas as pd
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# for iris dataset
'''iris = load_iris()
data = iris.feature_names
df = pd.DataFrame(iris.data, columns=data)
df['target'] = iris.target
print(df.head())

# separating flower types
flower1 = df[df.target == 0]
flower2 = df[df.target == 1]
flower3 = df[df.target == 2]

# visualization
plt.title('Flower distribution according to sepal length and width')
plt.xlabel('sepal length in (cm)')
plt.ylabel('sepal width (cm)')
plt.scatter(flower1['sepal length (cm)'],flower1['sepal width (cm)'],color='green')
plt.scatter(flower2['sepal length (cm)'],flower2['sepal width (cm)'],color='orange')
# plt.scatter(flower3['sepal length (cm)'],flower3['sepal width (cm)'],color='blue')
# plt.show()

# extract x and y
x = df.iloc[:,0:4].values
y = df.iloc[:,-1].values'''

# for digit dataset
digit = load_digits()
df = pd.DataFrame(digit.data,columns=digit.feature_names)
df['target'] = digit.target
print(df.head())

# extracting x and y
x = df.iloc[:,0:65].values
y = df.iloc[:,-1].values



# splitting data set
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=10)

# creating model object
model = SVC(gamma=10,kernel='linear')
model.fit(x_train,y_train)

# predicting
y_predict = model.predict(x_test)
print(y_predict)

# model accuracy
# testing 97
print(model.score(x_test,y_test)*100)