import numpy as np
import pandas as pd

# loading dataset
df = pd.read_csv('Dataset.csv')
import matplotlib.pyplot as plt

df.head()
df.describe()
df.info()
df.tail()
df.isna().sum()