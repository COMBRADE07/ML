import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from myfunction import Extracttion
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge

# load dataset
df = pd.read_csv('MagicBricks.csv')

print(df.head())
print(df.columns)
ext = Extracttion()

df['Locality'] = df['Locality'].apply(ext.extract_loc)
df['Locality'] = df['Locality'].apply(ext.remove_nums)
print(df['Locality'])