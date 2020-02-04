from numpy import array
import numpy as np

def split_sequence2(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence):
            break
        seq_x, seq_y = sequence[i:end_ix, :-1], sequence[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

in_seq1 = array([10,20,30,40,50,60,70,80,90,100])
in_seq2 = array([15,25,35,45,55,65,75,85,95,105])
print(in_seq1.info())

out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

print(in_seq1.shape)
print(out_seq.shape)

in_seq1 = in_seq1.reshape(len(in_seq1), 1)
in_seq2 = in_seq2.reshape(len(in_seq2), 1)
out_seq = out_seq.reshape(len(out_seq), 1)

#hstack
from numpy import hstack
dataset = hstack((in_seq1, in_seq2, out_seq)) #hstack: 10,1 3개가 합쳐져서 10,3
n_steps = 3
# print(dataset)

x, y = split_sequence2(dataset, n_steps)

for i in range(len(x)):
    print(x[i], y[i])

print(x.shape) # (8,3,2)
print(y.shape) # (8,)


x = x.reshape(8,6,1)
# DNN 지표는 loss | [[90 95] [100 105][110 115]]

from keras.models import Sequential
from keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(10, activation = 'relu', input_shape=(6,1))) # input_shape(열, 몇 개씩 자르는지) #(3,1): 열이 3개고 데이터 셋을 1개씩 잘라서 작업
model.add(Dense(5))                                         # 1개씩 자르면 결과는 잘 나옴 but 느림 <-> 2개씩 자르면 빠르지만 결과에 영향 << 하이퍼 파라미터 수정
model.add(Dense(1))

model.summary()

model.compile(loss='mae', optimizer='Adam', metrics=['mae'])
model.fit(x,y,epochs=100, batch_size=1) 
    
#4. 평가 예측
loss, mae = model.evaluate(x, y, batch_size=1) #3.test
print('mae:' , mae)
print('loss:' , loss)


x_prd = np.array([[90,95],[100,105],[110,115]])
# print(x_prd.shape)
x_prd = x_prd.reshape(1,6,1)
aaa = model.predict(x_prd, batch_size=1)
print(aaa)