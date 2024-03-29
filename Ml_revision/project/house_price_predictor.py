# importing library
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from myfunction import Extracttion, Encoder
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge

# load dataset
df = pd.read_csv('MagicBricks.csv')

# step 1. understand the data
print(df.head())
df.columns
df.shape
df.describe()
df.groupby('Furnishing')['Status'].agg('count')

# step 2. drop unnecessary columns
df = df.drop(['Parking', 'Status', 'Transaction', 'Type', 'Per_Sqft'], axis=1)

# step 3. Checking for Null values and fill them with mean
print(df.isna().sum())
mean_bathroom = df.Bathroom.mean()
# filling bathroom value with mean
df['Bathroom'] = df['Bathroom'].fillna(mean_bathroom)
# print(df.Furnishing.value_counts())
df['Furnishing'] = df.Furnishing.fillna(method='bfill')
# print(df.isna().sum())
df1 = df.copy()
df1.isna().sum()

# step 4. Performing necessary operation on data
df1.groupby('Locality')['Locality'].agg('count')


# this function will extract exact location
def extract_loc(loc):
    tokens = loc.split(' ')
    if len(tokens) > 3:
        return ' '.join(tokens[0:3])  # Join the extracted tokens back into a single string
    else:
        return loc


# this function will remove number from locality string
def remove_nums(str):
    if str:
        text = re.findall(r'[A-Za-z]+', str)
        t1 = ' '.join(text)
        return t1


# extracting propper locations
df1['Locality'] = df1['Locality'].apply(extract_loc)

# removing numbers from locality
df1['Locality'] = df1['Locality'].apply(remove_nums)

# replace locality count whoes count is less than 5.
loc_summarry = df1.groupby('Locality')['Locality'].agg('count').sort_values(ascending=False)
lessthan = loc_summarry[loc_summarry <= 5]
ll = df1.Locality.value_counts()

# df1.columns
# print(df1.isna().sum())

# update dataframe which has less than 5 count
df.Locality = df.Locality.apply(lambda x: 'other' if x in lessthan else x)

# checking for outliers
columns = ['Area', 'BHK', 'Bathroom', 'Price']


def check_outliers(columns):
    for column in columns:
        plt.subplot(3, 2, df.columns.get_loc(column) + 1)
        sns.boxplot(data=df[column])
        plt.title(f'Box Plot for {column}')
    plt.show()


# check_outliers(columns)
# fixing outlier

def remove_outliers(df):
    df1 = df.select_dtypes(include=['number'])  # for numbers
    df2 = df.select_dtypes(exclude=['number'])  # categoricalcolumns
    #  here we used IQR method
    ll = 0.25
    ul = 0.75

    Q1 = df1.quantile(q=ll)
    Q3 = df1.quantile(q=ul)
    IQR = Q3 - Q1

    lb = Q1 - 1.5 * IQR
    ub = Q3 + 1.5 * IQR
    df1 = df1[(df1 >= lb) & (df1 <= ub)]
    df = pd.concat([df1, df2], axis=1)
    return df


df = remove_outliers(df1)
df1.head()

# Encoding categorical data
oneHotEncoder = Encoder()
df2 = oneHotEncoder.onehot_encoder(df1,'Locality','Furnishing')
print(df2.head())
df3 = df2.copy()
# df3['Area','BHK','Bathroom'] = df1['Area','BHK','Bathroom']
print(df3.columns)
print(df3.isna().sum())

# extracting x and y
x = df1.drop('Price', axis='columns')
y = df1.Price
#
# # splitting data into training and testing
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)
print(len(x_train))
print(x_train)


lr = LinearRegression()
lr.fit(x_train,y_train)
print(lr.score(x_test,y_test))
#
