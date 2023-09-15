import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer

# load dataset
df = pd.read_csv('spam.csv')

# EDA
print(df.head())
print(df.columns)
print(df.isna().sum())
print(df.info())
print(df.groupby('Category').describe())


df['target'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
df = df.drop(['Category'], axis=1)
x = df.iloc[:, 0].values
y = df.iloc[:, 1].values

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)

# without pipeline
'''# count vectorizer
vect = CountVectorizer()
x_train_count = vect.fit_transform(x_train)
# x_train_count = x_train_count.toarray()

model = MultinomialNB()
model.fit(x_train_count,y_train)
x_test_count = vect.transform(x_test)
predicted = model.predict(x_test_count)
print(df['Message'].head(3))
print(predicted)
print(model.score(x_test_count,y_test))'''

# with pipeline
line = Pipeline([
    ('vecto',CountVectorizer()),
    ('mul',MultinomialNB())
])
line.fit(x_train,y_train)
print(line.score(x_test,y_test))