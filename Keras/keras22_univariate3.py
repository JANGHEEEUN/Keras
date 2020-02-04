from numpy import array
import numpy as np

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

dataset = [10,20,30,40,50,60,70,80,90,100]

n_steps = 3

x, y = split_sequence(dataset, n_steps)

for i in range(len(x)):
    print(x[i], y[i])


print(x.shape) #(7,3)
print(y.shape) #(7,)

y.reshape(7,1)
x = x.reshape(x.shape[0], x.shape[1], 1)

from keras.models import Sequential
from keras.layers import Dense, Input, LSTM

model = Sequential()
model.add(LSTM(10, activation = 'relu', input_shape=(3,1))) # input_shape(열, 몇 개씩 자르는지) #(3,1): 열이 3개고 데이터 셋을 1개씩 잘라서 작업
model.add(Dense(5))                                         # 1개씩 자르면 결과는 잘 나옴 but 느림 <-> 2개씩 자르면 빠르지만 결과에 영향 << 하이퍼 파라미터 수정
model.add(Dense(1))
model.summary()

model.compile(loss='mae', optimizer='Adam', metrics=['mae'])
model.fit(x,y,epochs=200, batch_size=1) 
    
#4. 평가 예측
loss, mae = model.evaluate(x, y, batch_size=1) #3.test
print('mae:' , mae)
print('loss:' , loss)


x_prd = np.array([90,100,110])
x_prd = x_prd.reshape(1,3,1)
print(x_prd.shape)
aaa = model.predict(x_prd, batch_size=1)
print(aaa)

0,1,2,3 / 4
1,2,3,4 / 5
