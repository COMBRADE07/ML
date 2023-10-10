import re

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
class Encoder:
    def onehot_encoder(self, df,c1,c2):
        encoder = OneHotEncoder(drop='first',sparse_output=False)
        data = encoder.fit_transform(df[[c1, c2]])
        col = encoder.get_feature_names_out([c1,c2])
        encoded_df = pd.DataFrame(data,columns=col)
        return encoded_df


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