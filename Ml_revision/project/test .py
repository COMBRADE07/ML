import re
import pandas as pd

def remove_nums(str):
    if str:
        text = re.findall(r'[A-Za-z]+', str)
        t1 = ' '.join(text)
        return t1



s = 'hello rhuy55ik this is, 35'
print(s.isalnum())
x = remove_nums(s)
print(x)