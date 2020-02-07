import numpy as np
import pandas as pd

samsung = np.load('./Keras/Downloads/data/samsung.npy')
kospi200 = np.load('./Keras/Downloads/data/kospi.npy')

x1_prd = samsung[426:430,:]
x2_prd = kospi200[421:426, :]
print(x1_prd)

def split_xy5(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :]
        tmp_y = dataset[x_end_number:y_end_number,3]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x1, y1 = split_xy5(samsung, 5, 1)
x2, y2 = split_xy5(kospi200, 5, 1)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, random_state=1, test_size=0.2, shuffle=False)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, random_state=1, test_size=0.2, shuffle=False)

x1_train = x1_train.reshape(x1_train.shape[0],x1_train.shape[1]*x1_train.shape[2])
x1_test = x1_test.reshape(x1_test.shape[0],x1_test.shape[1]*x1_test.shape[2])
x2_train = x2_train.reshape(x2_train.shape[0],x2_train.shape[1]*x2_train.shape[2])
x2_test = x2_test.reshape(x2_test.shape[0],x2_test.shape[1]*x2_test.shape[2])


#데이터 전처리
#StandardScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = StandardScaler()
scaler.fit(x1_train)
x1_train_scaled = scaler.transform(x1_train)
x1_test_scaled = scaler.transform(x1_test)

scaler = StandardScaler()
scaler.fit(x2_train)
x2_train_scaled = scaler.transform(x2_train)
x2_test_scaled = scaler.transform(x2_test)

x1_train_scaled = x1_train_scaled.reshape(x1_train_scaled.shape[0],25,1)
x1_test_scaled = x1_test_scaled.reshape(x1_test_scaled.shape[0],25,1)
x2_train_scaled = x2_train_scaled.reshape(x2_train_scaled.shape[0],25,1)
x2_test_scaled = x2_test_scaled.reshape(x2_test_scaled.shape[0],25,1)

scaler.fit(x1_prd)
x1_prd_scaled = scaler.transform(x1_prd)
x1_prd_scaled = x1_prd_scaled.reshape(1,25,1)

scaler.fit(x2_prd)
x2_prd_scaled = scaler.transform(x2_prd)
x2_prd_scaled = x2_prd_scaled.reshape(1,25,1)


print(x1_train_scaled)
print(x1_prd_scaled)
print(x1_train_scaled.shape)
print(x1_prd_scaled.shape)

from keras import layers
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input

input_1 = layers.Input(shape=(25,1))
input_tensor_1 = layers.LSTM(64, activation='relu')(input_1) 
hidden_layer_1 = layers.Dense(128, activation='relu')(input_tensor_1)
hidden_layer_1 = layers.Dense(64, activation='relu')(hidden_layer_1)
output_tensor_1 = layers.Dense(16, activation='relu')(hidden_layer_1)

input_2 = layers.Input(shape=(25,1))
input_tensor_2 = layers.LSTM(128, activation='relu')(input_2) 
hidden_layer_2 = layers.Dense(16, activation='relu')(input_tensor_2)
hidden_layer_2 = layers.Dense(64,activation='relu')(hidden_layer_2)
output_tensor_2 = layers.Dense(28, activation='relu')(hidden_layer_2)

from keras.layers.merge import concatenate , Add

merged_model = concatenate(inputs=[output_tensor_1, output_tensor_2]) 
output_tensor_3 = layers.Dense(16, activation='relu')(merged_model)   
output_tensor_3 = layers.Dense(1)(output_tensor_3)

model = Model(inputs=[input_1, input_2],
              outputs=output_tensor_3)

model.compile(loss='mae', optimizer='Adam', metrics=['mae'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience=20)

model.fit([x1_train_scaled, x2_train_scaled], y1_train, epochs=1000, batch_size=10, validation_split=0.2)
#validation_split: val data 없이 validation을 하고 싶다면 train 데이터에서 20%를 val로 할당 

loss, mae = model.evaluate([x1_test_scaled, x2_test_scaled], y1_test, batch_size=5) #3.test
print('mae:' , mae)
print('loss:' , loss)

y_prd = model.predict([x1_prd_scaled, x2_prd_scaled])
print(y_prd)

'''
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
    #np.sqrt: 루트
print("RMSE : ", RMSE(y1_test, y_prd))
'''