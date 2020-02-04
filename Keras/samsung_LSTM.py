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

# print(data.head())

# print(data.shape)

data1 = data.iloc[:, 1:6]


data1.rename(columns = {'2':'0', '3':'1', '4':'2', 
                       '5':'3', '6':'4'}, inplace=True)


data1['0'] = data1['0'].str.replace(pat=r'[^A-Za-z0-9]', repl=r'', regex=True)
data1['1'] = data1['1'].str.replace(pat=r'[^A-Za-z0-9]', repl=r'', regex=True)
data1['2'] = data1['2'].str.replace(pat=r'[^A-Za-z0-9]', repl=r'', regex=True)
data1['3'] = data1['3'].str.replace(pat=r'[^A-Za-z0-9]', repl=r'', regex=True)
data1['4'] = data1['4'].str.replace(pat=r'[^A-Za-z0-9]', repl=r'', regex=True)

df1 = pd.DataFrame(data1)
df1 = df1.apply(pd.to_numeric)
print(df1.info())


df1 = np.array(df1)

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
X, y = split_sequence(df1, n_steps)

print(data1.head(10))
for i in range(5):
    print(X[i], y[i])
    
print(X.shape) #(421, 5,5)
print(y.shape) #(421,)

X = X.reshape(421, 25, 1)

from keras.models import Sequential
from keras.layers import Dense, LSTM

model = Sequential()

model.add(LSTM(10, activation = 'relu', input_shape=(25,1))) # input_shape(열, 몇 개씩 자르는지) #(3,1): 열이 3개고 데이터 셋을 1개씩 잘라서 작업
model.add(Dense(5))                                         # 1개씩 자르면 결과는 잘 나옴 but 느림 <-> 2개씩 자르면 빠르지만 결과에 영향 << 하이퍼 파라미터 수정
model.add(Dense(1))

model.summary()

model.compile(loss='mae', optimizer='Adam', metrics=['mae'])
model.fit(X,y,epochs=100, batch_size=1) 

loss, mae = model.evaluate(X, y, batch_size=1) #3.test
print('mae:' , mae)
print('loss:' , loss)

x_prd = X[0:1,:]
print(x_prd.shape)
x_prd.reshape(1,25,1)
print(x_prd)
aaa = model.predict(x_prd, batch_size=1)
print(aaa)

y_predict = model.predict(X, batch_size=1)
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y, y_predict))
    #np.sqrt: 루트
print("RMSE : ", RMSE(y, y_predict))