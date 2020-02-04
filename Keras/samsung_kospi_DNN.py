#col 5개 DNN 구성... 컬럼이 5개인 종가만 구하면 됨

import pandas as pd
from pandas import DataFrame
import numpy as np
from numpy import array

sam = pd.read_csv('Downloads/samsung.csv',  encoding = 'ISO-8859-1')

sam.rename(columns = {'ÀÏÀÚ':'1', '½Ã°¡':'2', '°í°¡':'3', 
                       'Àú°¡':'4', 'Á¾°¡':'5', '°Å·¡·®':'6'}, inplace=True)

sam = sam.iloc[:, 1:6]
sam.rename(columns = {'2':'0', '3':'1', '4':'2', 
                       '5':'3', '6':'4'}, inplace=True)


sam['0'] = sam['0'].str.replace(pat=r'[^A-Za-z0-9]', repl=r'', regex=True)
sam['1'] = sam['1'].str.replace(pat=r'[^A-Za-z0-9]', repl=r'', regex=True)
sam['2'] = sam['2'].str.replace(pat=r'[^A-Za-z0-9]', repl=r'', regex=True)
sam['3'] = sam['3'].str.replace(pat=r'[^A-Za-z0-9]', repl=r'', regex=True)
sam['4'] = sam['4'].str.replace(pat=r'[^A-Za-z0-9]', repl=r'', regex=True)

df_sam = pd.DataFrame(sam)
df_sam = df_sam.apply(pd.to_numeric)


ko = pd.read_csv('Downloads/kospi200.csv',  encoding = 'ISO-8859-1')

# print(ko.info())

ko.rename(columns = {'ÀÏÀÚ':'1', '½Ã°¡':'2', '°í°¡':'3', 
                       'Àú°¡':'4', 'ÇöÀç°¡':'5', '°Å·¡·®':'6'}, inplace=True)
# print(ko.info())

ko = ko.iloc[:, 1:6]
ko.rename(columns = {'2':'0', '3':'1', '4':'2', 
                       '5':'3', '6':'4'}, inplace=True)


ko['4'] = ko['4'].str.replace(pat=r'[^A-Za-z0-9]', repl=r'', regex=True)

df_ko = pd.DataFrame(ko)
df_ko = df_ko.apply(pd.to_numeric)
# print(df_ko.info())

df_sam = np.array(df_sam)
df_ko = np.array(df_ko)


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix, -2]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

n_steps = 5
x, y = split_sequence(df_sam, n_steps)
x1, y1 = split_sequence(df_ko, n_steps)

# for i in range(5):
#     print(x[i], y[i])

# for i in range(5):
#     print(x1[i], y1[i])
    
print(x.shape) #(421, 5,5)
print(y.shape) #(421,)
print(x1.shape) #(421, 5,5)
print(y1.shape) #(421,)

x = x.reshape(421, 25)
x1 = x1.reshape(421, 25)

from keras.models import Sequential, Model 
from keras.layers import Dense, Input
input1 = Input(shape=(25,))
dense1 = Dense(5)(input1)
dense2 = Dense(2)(dense1)
dense3 = Dense(3)(dense2)
output1 = Dense(3)(dense3) 

input2 = Input(shape=(25,))
dense21 = Dense(5)(input2)
dense22 = Dense(2)(dense21)
dense23 = Dense(3)(dense22)
output2 = Dense(3)(dense23)

from keras.layers.merge import Concatenate 
merge1 = Concatenate()([output1, output2])

middle1 = Dense(3)(merge1)
middle2 = Dense(7)(middle1)
output = Dense(1)(middle2)

model = Model(inputs = [input1, input2], outputs=output)

model.compile(loss='mse', optimizer='Adam', metrics=['mse'])
model.fit([x, x1], y,epochs=100, batch_size=10)

loss, mae = model.evaluate([x,x1],y, batch_size=1) #3.test
print('mae:' , mae)
print('loss:' , loss)

x_prd = x[0:1,:]
x1_prd = x1[0:1,:]
print(x_prd.shape)
aaa = model.predict([x_prd, x1_prd], batch_size=1)
print(aaa)

y_predict = model.predict([x, x1], batch_size=1)
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y, y_predict))
    #np.sqrt: 루트
print("RMSE : ", RMSE(y, y_predict))