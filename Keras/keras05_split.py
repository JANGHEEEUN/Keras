import numpy as np 
#keras04_val.py 문제점: 데이터가 train, test, validation으로 나눠지지 않음 -> split!

#1. 데이터 - 전처리를 할 필요가 없는 정제된 데이터
x = np.array(range(1,101))
y = np.array(range(1,101))

x_train = x[:60]
y_train = y[:60]
x_test = x[60:80]
y_test = y[60:80]
x_val = x[80:100]
y_val = y[80:100]


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
model.fit(x_train,y_train,epochs=100, batch_size=1, validation_data=(x_val,y_val)) #1.train -> 2.val  
                                                                                    # val: 정확도를 높이는데 큰 영향
    
#4. 평가 예측
loss, mae = model.evaluate(x_test,y_test, batch_size=1) #3.test
print('mae:' , mae)
print('loss:' , loss)


x_prd = np.array([201,202,203])
aaa = model.predict(x_prd, batch_size=1)
print(aaa)
