import numpy as np 

#1. 데이터
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15,16,17,18,19,20])
y_test = np.array([11,12,13,14,15,16,17,18,19,20])

# print(x.shape) 
# print(y.shape)

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(5, input_shape = (1, )))
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(1))

model.summary()

#3. 훈련
model.compile(loss='mse', optimizer='Adam', metrics=['mae'])
model.fit(x_train,y_train,epochs=100, batch_size=1)
    
#4. 평가 예측
loss, mae = model.evaluate(x_test,y_test, batch_size=1)
print('mae:' , mae)
print('loss:' , loss)

x_prd = np.array([21,22,23])
aaa = model.predict(x_prd, batch_size=1)
print(aaa)
