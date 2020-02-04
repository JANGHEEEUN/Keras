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

x = x.reshape(421, 25,1)
x1 = x1.reshape(421, 25,1)

from keras import layers
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input

input_1 = layers.Input(shape=(25,1))
input_tensor_1 = layers.LSTM(32, activation='relu')(input_1)   # x1에 투입할 모델
hidden_layer_1 = layers.Dense(16, activation='relu')(input_tensor_1)
hidden_layer_1 = layers.Dense(8, activation='relu')(hidden_layer_1)
output_tensor_1 = layers.Dense(1, activation='relu')(hidden_layer_1)

input_2 = layers.Input(shape=(25,1))
input_tensor_2 = layers.LSTM(32, activation='relu')(input_2)    # x2에 투입할 모델
hidden_layer_2 = layers.Dense(16, activation='relu')(input_tensor_2)
hidden_layer_2 = layers.Dense(8, activation='relu')(hidden_layer_2)
output_tensor_2 = layers.Dense(1, activation='relu')(hidden_layer_2)

from keras.layers.merge import concatenate , Add

merged_model = concatenate(inputs=[output_tensor_1, output_tensor_2])  # 두 개 이상은 []로 묶기
# merged_model = Add()([output_tensor_1, output_tensor_2])  # concatenate 대신 Add를 사용해도 가능하다.

output_tensor_3 = layers.Dense(8)(merged_model)        # 첫 번째 아웃풋 모델
output_tensor_3 = layers.Dense(1)(output_tensor_3)

# output_tensor_4 = layers.Dense(8)(merged_model)        # 두 번째 아웃풋 모델
# output_tensor_4 = layers.Dense(1)(output_tensor_4)

model = Model(inputs=[input_1, input_2],
              outputs=output_tensor_3)

model.compile(loss='mse', optimizer='Adam', metrics=['mse'])
model.fit([x, x1], y,epochs=200, batch_size=1, verbose=99)

loss= model.evaluate([x,x1],y, batch_size=1) #3.test
print('loss:' , loss)

x_prd =x[0:1,:,:]
x1_prd =x1[0:1,:,:]


x_prd.reshape(1,25,1)
x1_prd.reshape(1,25,1)

aaa = model.predict([x_prd, x1_prd])
print(aaa)

y_predict = model.predict([x, x1], batch_size=1)
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y, y_predict))
    #np.sqrt: 루트
# print("RMSE : ", (RMSE(y, y_predict[0])+RMSE(y1, y_predict[1]))/2)
print("RMSE : ", RMSE(y, y_predict))