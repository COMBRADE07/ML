import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# load dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data,columns=iris.feature_names)
df['target'] = iris.target

# flower 1 2 3
flower1 = df[0:50]
flower2 = df[50:100]
flower3 = df[100:]

print(flower1.head())
# visualization
plt.scatter(x=flower1['sepal length (cm)'],y=flower1['sepal width (cm)'],color='green',marker='*')
plt.scatter(x=flower2['sepal length (cm)'],y=flower2['sepal width (cm)'],color='blue')
# plt.scatter(x=flower3['sepal length (cm)'],y=flower3['sepal width (cm)'],color='orange')

plt.title('flower categorization by sepal wise')
plt.xlabel('sepal length in (cm)')
plt.ylabel('sepal width (cm)')
# plt.legend()
# plt.show()

x = df.iloc[:,4].values
y = df.iloc[:,-1].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)

# model
model = KNeighborsClassifier(n_neighbors=50)
model.fit(x_train.reshape(-1, 1),y_train)
print(model.score(x_test.reshape(-1, 1),y_test))

# confusion matrix
from sklearn.metrics import confusion_matrix

y_pred = model.predict(x_test.reshape(-1,1))
cm = confusion_matrix(y_test.reshape(-1,1),y_pred)
print(cm)
