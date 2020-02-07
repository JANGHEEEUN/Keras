import numpy as np
import pandas as pd

samsung = np.load('./Keras/Downloads/samsung2.npy')
kospi200 = np.load('./Keras/Downloads/data/kospi.npy')
x1_prd = samsung[425:430,:]
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

x, y = split_xy5(samsung, 5, 1)

# print(x.shape) #(421,5,5)
# print(y.shape) #(421,1)
# print(x[0,:], '\n', y[0])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.3, shuffle=False)
print(x_train.shape)
print(x_test.shape)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])

#데이터 전처리
#StandardScaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

scaler.fit(x1_prd)
x1_prd_scaled = scaler.transform(x1_prd)
x1_prd_scaled = x1_prd_scaled.reshape(1,25)

from keras.models import Sequential
from keras.layers import Dense, Input

model = Sequential()

model.add(Dense(25, input_shape =(25, )))
model.add(Dense(16))
model.add(Dense(4))
model.add(Dense(1))

model.summary()

model.compile(loss='mae', optimizer='Adam', metrics=['mae'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience=20)

model.fit(x_train_scaled, y_train,epochs=200, batch_size=5, validation_split=0.2, callbacks=[early_stopping])
#validation_split: val data 없이 validation을 하고 싶다면 train 데이터에서 20%를 val로 할당 

loss, mae = model.evaluate(x_test_scaled, y_test, batch_size=5) #3.test
print('mae:' , mae)
print('loss:' , loss)

y_prd = model.predict(x1_prd_scaled)
print(y_prd)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
    #np.sqrt: 루트
print("RMSE : ", RMSE(y_test, y_prd))
