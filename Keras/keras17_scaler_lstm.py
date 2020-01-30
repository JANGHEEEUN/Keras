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

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

model = Sequential()
model.add(LSTM(10, activation = 'relu', input_shape=(3,1))) # input_shape(열, 몇 개씩 자르는지) #(3,1): 열이 3개고 데이터 셋을 1개씩 잘라서 작업
model.add(Dense(7)) 
model.add(Dense(4))                                      
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='Adam', metrics=['mae']) # metrics mae = 반환값 2개
model.fit(x_train, y_train, epochs=300, batch_size=1) 

loss, mae = model.evaluate(x_test, y_test, batch_size=1)

print('loss: ' , loss) # mse 출력
print('mae: ', mae)

x_input = array([250, 260, 270]) # (3,) -> (1, 3) -> (1 , 3, 1) 전체를 곱한 값이 같기 때문에 reshape 가능
x_input = x_input.reshape(1,3,1)

y_predict = model.predict(x_input)
print(y_predict)

from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2: ", r2_y_predict)
