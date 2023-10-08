import re

from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
class Encoder:
    def onehot_encoder(self, df):
        encoder = OneHotEncoder()
        x = encoder.fit_transform(df[['Furnishing', 'Locality']])
        return x


class Outlier:
    def check_outliers(self,df,columns):
        for column in columns:
            plt.subplot(3, 2, df.columns.get_loc(column) + 1)
            sns.boxplot(data=df[column])
            plt.title(f'Box Plot for {column}')
        plt.show()

class Extracttion:
    def remove_nums(self,str):
        if str:
            text = re.findall(r'[A-Za-z]+', str)
            t1 = ' '.join(text)
            return t1

    def extract_loc(self,loc):
        tokens = loc.split(' ')
        if len(tokens) > 3:
            return ' '.join(tokens[0:3])  # Join the extracted tokens back into a single string
        else:
            return loc