import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris

# loading and reading dataset
iris = load_iris()
col = iris.feature_names
df = pd.DataFrame(iris.data,columns=col)


# in this assignment we have to form cluster using petal length and width
# dropping remaining features
df = df.drop(['sepal length (cm)','sepal width (cm)'],axis=1)
print(df.head())

# EDA
print(df.info())
print(df.describe())
print(df.isna().sum())

# checking for outliers
for i in df.columns:
    plt.subplot(2, 2, df.columns.get_loc(i) + 1)
    sns.boxplot(data=df[i])
    plt.title(f'Box plot for {i}')
plt.show()

'''
    conclusion
    - their is no outlier in given dataset
    - everything looks fine
    - we can go ahead
'''

# scaling data
scaler = MinMaxScaler()
df = scaler.fit_transform(df) # this will return ndarray as output
df = pd.DataFrame(data=df)
# print(df.head())

# finding best value for k using elbow method
intertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,random_state=0,n_init=10)
    kmeans.fit(df)
    intertia.append(kmeans.inertia_)

print(intertia)

# visualization of elbow plot
plt.title("Elbow plot")
plt.plot(range(1,11),intertia)
plt.xlabel("K values")
plt.ylabel("sum of squared errors")
plt.show()

'''
    elbow plot conclusion
    - we got k value for above elbow plot is 2
'''

# building model
model = KMeans(n_clusters=2,n_init=10,random_state=0)
model.fit_transform(df)
y = model.predict(df)

df['clusters'] = y

# drawing cluster scatter plots
df1 = df[df['clusters'] == 0]
df2 = df[df['clusters'] == 1]

plt.scatter(df1.iloc[:,0],df1.iloc[:,1],color="green")
plt.scatter(df2.iloc[:,0],df2.iloc[:,1],color="red")
plt.show()

'''
    from above scatterplot,
    - without scaling
    - their is one cluster value is miss leading 
    
    = with scaling
      - problem solved 
'''