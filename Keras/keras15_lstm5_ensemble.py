
#1. 데이터
from numpy import array
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input

x1 = array([[1,2,3], [2,3,4],[3,4,5], [4,5,6], 
           [5,6,7], [6,7,8], [7,8,9], [8,9,10],
           [9,10,11], [10,11,12], [20,30,40], [30,40,50], [40,50,60]]) #(13,3)
y1 = array([4,5,6,7,8,9,10,11,12,13,50,60,70]) # (13,) - 벡터

x2 = array([[10,20,30], [20,30,40],[30,40,50], [40,50,60], 
           [50,60,70], [60,70,80], [70,80,90], [80,90,100],
           [90,100,110], [100,110,120], [2,3,4], [3,4,5], [4,5,6]]) #(13,3)
y2 = array([40,50,60,70,80,90,100,110,120,130,5,6,7]) # (13,) - 벡터

x1 = x1.reshape(x1.shape[0], x1.shape[1], 1) # shape: (13,3,1)
x2 = x2.reshape(x2.shape[0], x2.shape[1], 1) # shape: (13,3,1)

# 2. 모델 구성
from keras.models import Sequential, Model 
from keras.layers import Dense, Input

input1 = Input(shape=(3,1))
model1 = LSTM(10, activation='relu')(input1)
dense1 = Dense(5)(model1)
dense2 = Dense(2)(dense1)
output1 = Dense(1)(dense2) 

input2 = Input(shape=(3,1))
model2 = LSTM(10, activation='relu')(input2)
dense21 = Dense(5)(model2)
dense22 = Dense(2)(dense21)
output2 = Dense(1)(dense22) 

from keras.layers.merge import concatenate
merge1 = concatenate([output1, output2])

middle1 = Dense(4)(merge1)
middle2 = Dense(7)(middle1)
middle3 = Dense(1)(middle2) #현재 merge된 마지막 레이어

output_1 = Dense(30)(middle3) #1번째 output 모델
output_1 = Dense(3)(output_1) #Dense(n)을 col과 맞추기

output_2 = Dense(30)(middle3) #3번째 output 모델
output_2= Dense(3)(output_2)

model = Model(inputs = [input1, input2], outputs=[output_1,output_2])
#model input이 2개이므로 리스트로 넣어줌

model.summary()


model.compile(loss='mse', optimizer='Adam', metrics=['mae']) # metrics mae = 반환값 2개
model.fit([x1,x2], [y1,y2], epochs=100, batch_size=1, verbose=99)  #verbose [default] = 1 - verbose는 fitting 진행 상황을 보여줌
#verbose 2: 막대 빼고 간결하게
#오래된 데이터라면 verbose0을 두는 것이 좋음


#4. 평가 예측
loss, mae = model.evaluate([x1,x2], [y1,y2], batch_size=1) #3.test

print('loss: ' , loss) # mse 출력
print('mae: ', mae)

x_input = array([[6.5,7.5,8.5], [50,60,70], [70,80,90], [100,110,120]]) # (3,) -> (1, 3) -> (1 , 3, 1) 전체를 곱한 값이 같기 때문에 reshape 가능
x_input = x_input.reshape(4,3,1)

y_predict = model.predict(x_input)
print(y_predict)
