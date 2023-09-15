import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# reading dataset
df = pd.read_csv('wineclustering.csv')

# EDA
print(df.head())
print(df.columns)
print(df.info())
print(df.isna().sum())

# checking for outliers
columns = ['Alcohol', 'Malic_Acid', 'Ash', 'Ash_Alcanity', 'Magnesium', 'Total_Phenols', 'Flavanoids',
           'Nonflavanoid_Phenols', 'Proanthocyanins', 'Color_Intensity', 'Hue', 'OD280', 'Proline']

for column in columns:
    plt.subplot(5, 3, df.columns.get_loc(column) + 1)
    sns.boxplot(data=df[column])
    plt.title(f'Box Plot for {column}')
plt.show()

'''
    conclusion
    - outliers present in given dataset
    - their is no empty column
    - all data in numeric formate so we don't need to transform 
    - this data is already cleaned 
'''

# removing outliers
q1 = df.quantile(0.25)
q3 = df.quantile(0.75)

IQR = q3 - q1
l1 = q1 - 1.5 * IQR
l2 = q3 - 1.5 * IQR

df = df[~((df < l1) | (df > l2)).any(axis=1)]



# model object
model = KMeans(n_clusters=2)

y = model.fit_predict(df[['Alcohol', 'Malic_Acid', 'Ash', 'Ash_Alcanity', 'Magnesium', 'Total_Phenols', 'Flavanoids',
           'Nonflavanoid_Phenols', 'Proanthocyanins', 'Color_Intensity', 'Hue', 'OD280', 'Proline']])
print(y)