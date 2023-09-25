import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge

# load dataset
df = pd.read_csv('MagicBricks.csv')

# step 1. understand the data
print(df.head())
print("columns: ", df.columns)
print("Shape of data: ", df.shape)
print(df.describe())
print(df.groupby('Furnishing')['Status'].agg('count'))

# step 2. drop unnecessary columns
df = df.drop(['Parking', 'Status', 'Transaction', 'Type', 'Per_Sqft'], axis=1)

# step 3. Checking for Null values and fill them with mean
print(df.isna().sum())
mean_bathroom = df.Bathroom.mean()
# filling bathroom value with mean
df['Bathroom'] = df['Bathroom'].fillna(mean_bathroom)
print(df.Furnishing.value_counts())
df.Furnishing = df.Furnishing.fillna(method='ffill')
print(df.isna().sum())


# step 4.
# print(df.groupby('Locality')['Locality'].agg('count'))

def extract_loc(loc):
    tokens = loc.split(' ')
    if len(tokens) > 3:
        return ' '.join(tokens[0:3])  # Join the extracted tokens back into a single string
    else:
        return loc


# extracting propper locations
df['Locality'] = df['Locality'].apply(extract_loc)

# group location
df_locations = df.groupby('Locality')['Locality'].agg('count')
