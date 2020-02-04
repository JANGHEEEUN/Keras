#col 5개 DNN 구성... 컬럼이 5개인 종가만 구하면 됨

import pandas as pd
from pandas import DataFrame
import numpy as np
from numpy import array

data = pd.read_csv('Downloads/samsung.csv',  encoding = 'ISO-8859-1')

data.rename(columns = {'ÀÏÀÚ':'1', '½Ã°¡':'2', '°í°¡':'3', 
                       'Àú°¡':'4', 'Á¾°¡':'5', '°Å·¡·®':'6'}, inplace=True)

data['1'] = data['1'].str[0:4] +  data['1'].str[5:7] +  data['1'].str[8:10]
data['2'] = data['2'].str[0:4] +  data['1'].str[5:7] +  data['1'].str[8:10]

print(data.head())

print(data.shape)

data1 = data.iloc[:, 1:6]
print(data1.head())

data1.rename(columns = {'2':'0', '3':'1', '4':'2', 
                       '5':'3', '6':'4'}


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[3]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


n_steps = 5
X, y = split_sequence(data1, n_steps)

for i in range(len(x)):
    print(x[i], y[i])