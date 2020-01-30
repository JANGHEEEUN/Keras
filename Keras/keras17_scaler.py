import numpy as np
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

x = array([[1,2,3], [2,3,4],[3,4,5], [4,5,6], 
           [5,6,7], [6,7,8], [7,8,9], [8,9,10],
           [9,10,11], [10,11,12], [20000,30000,40000], 
           [30000,40000,50000], [40000,50000,60000], [100,200,300]]) #(13,3)
y = array([4,5,6,7,8,9,10,11,12,13,50000,60000,70000,400]) # (13,) - 벡터

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import RobustScaler, MaxAbsScaler

x_train = x[:10]
x_test = x[10:14]
y_train = y[:10]
y_test = y[10:14]

#사용법: 정의 -> fitting -> transform
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_train)

#train 10, 나머지는 test - no validation
#dense 모델로 구현
x_train = x[:10]
x_test = x[10:14]
y_train = y[:10]
y_test = y[10:14]
#Dense 모델 구현
#R2 지표

from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(5, input_shape = (3,)))
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(1))

model.compile(loss='mse', optimizer='Adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=1000, batch_size=1)

loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print('mse, loss:  ', mse, loss)

x_prd = np.array([[250, 260, 270]])
aaa = model.predict(x_prd, batch_size=1)
print(aaa)