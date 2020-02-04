import numpy as np
import pandas as pd

samsung = np.load('Downloads/data/samsung.npy')
kospi200 = np.load('Downloads/data/kospi.npy')

# print(samsung)
# print(samsung.shape)
# print(kospi200)
# print(kospi200.shape)

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

# print(x.shape) #(421,5,5)
# print(y.shape) #(421,1)
# print(x[0,:], '\n', y[0])

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, random_state=1, test_size=0.3, shuffle=False)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, random_state=1, test_size=0.3, shuffle=False)

# x1_train = np.reshape(x1_train, x1_train.shape[0],x1_train.shape[1]*x1_train.shape[2])
# x1_test = np.reshape(x1_test, x1_test.shape[0],x1_test.shape[1]*x1_test.shape[2])
# x2_train = np.reshape(x2_train, x2_train.shape[0],x2_train.shape[1]*x2_train.shape[2])
# x2_test = np.reshape(x2_test, x2_test.shape[0],x2_test.shape[1]*x2_test.shape[2])

x1_train = x1_train.reshape(x1_train.shape[0],x1_train.shape[1]*x1_train.shape[2])
x1_test = x1_test.reshape(x1_test.shape[0],x1_test.shape[1]*x1_test.shape[2])
x2_train = x2_train.reshape(x2_train.shape[0],x2_train.shape[1]*x2_train.shape[2])
x2_test = x2_test.reshape(x2_test.shape[0],x2_test.shape[1]*x2_test.shape[2])


#데이터 전처리
#StandardScaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x1_train)
x1_train_scaled = scaler.transform(x1_train)
x1_test_scaled = scaler.transform(x1_test)

scaler = StandardScaler()
scaler.fit(x2_train)
x2_train_scaled = scaler.transform(x2_train)
x2_test_scaled = scaler.transform(x2_test)

# print(x1_train_scaled[0,:])


from keras.models import Sequential, Model
from keras.layers import Dense, Input

input1 = Input(shape=(25,))
dense1 = Dense(52)(input1)
dense2 = Dense(22)(dense1)
dense3 = Dense(23)(dense2)
output1 = Dense(32)(dense3) 

input2 = Input(shape=(25,))
dense21 = Dense(52)(input2)
dense22 = Dense(22)(dense21)
dense23 = Dense(32)(dense22)
output2 = Dense(32)(dense23)

from keras.layers.merge import Concatenate 
merge1 = Concatenate()([output1, output2])

middle1 = Dense(3)(merge1)
middle2 = Dense(7)(middle1)
output = Dense(1)(middle2)

model = Model(inputs = [input1, input2], outputs=output)

model.compile(loss='mae', optimizer='Adam', metrics=['mae'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience=20)

model.fit([x1_train_scaled, x2_train_scaled], y1_train, epochs=200, batch_size=5, validation_split=0.2, callbacks=[early_stopping])
#validation_split: val data 없이 validation을 하고 싶다면 train 데이터에서 20%를 val로 할당 

loss, mae = model.evaluate([x1_test_scaled, x2_test_scaled], y1_test, batch_size=5) #3.test
print('mae:' , mae)
print('loss:' , loss)

y_prd = model.predict([x1_test_scaled, x2_test_scaled])
print(y_prd)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
    #np.sqrt: 루트
print("RMSE : ", RMSE(y1_test, y_prd))
