
#1. 데이터
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM, Reshape

x = array([[1,2,3], [2,3,4],[3,4,5], [4,5,6], 
           [5,6,7], [6,7,8], [7,8,9], [8,9,10],
           [9,10,11], [10,11,12], [20,30,40], [30,40,50], [40,50,60]]) #(13,3)
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70]) # (13,) - 벡터

x = x.reshape(x.shape[0], x.shape[1], 1) # x를 (5,3,1)로 reshape - 뒤에 몇 개씩 자르는지 붙여줘야 함
                                         # 5,3,1은 곱했을 때 원 데이터와 같이 15가 나옴

# 2. 모델 구성
model = Sequential()
model.add(LSTM(10, activation = 'relu', return_sequences=True, input_shape=(3,1))) # input_shape(열, 몇 개씩 자르는지) #(3,1): 열이 3개고 데이터 셋을 1개씩 잘라서 작업
model.add(LSTM(2, activation = 'relu', return_sequences=True)) #input(n,3,1)의 데이터가 시계열이라는 확신이 없음
model.add(LSTM(3, activation = 'relu', return_sequences=True)) # (n,3,2) 역시 시계열일는 확신 없음
model.add(LSTM(4, activation = 'relu', return_sequences=True)) 
model.add(LSTM(5, activation = 'relu', return_sequences=True)) 
model.add(LSTM(6, activation = 'relu', return_sequences=True)) 
model.add(LSTM(7, activation = 'relu', return_sequences=True)) 
model.add(LSTM(8, activation = 'relu', return_sequences=True)) 
model.add(LSTM(9, activation = 'relu', return_sequences=True)) 
model.add(LSTM(10, activation = 'relu', return_sequences=False)) 
model.add(Dense(5, activation='linear'))                                        
model.add(Dense(1))
# model.add(LSTM(20, activation = 'relu', return_sequences=False)) #return_sequences false가 default

#데이터가 백만개여도 lstm을 많이 쓰면 성능이 안 좋아짐
#Cnn은 여러번 엮을수로 더 좋음 하지만 시계열은 여러번 엮을수록 데이터가 점점 더 시계열이라는 확신을 잃어버리므로 사용하지 않음


model.summary()

#3. 모델 훈련
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto') #monitor: loss | acc | val_loss | val_acc

model.compile(loss='mse', optimizer='Adam', metrics=['mae']) # metrics mae = 반환값 2개
model.fit(x, y, epochs=1000, batch_size=1, verbose=1, callbacks=[early_stopping])  #verbose [default] = 1 - verbose는 fitting 진행 상황을 보여줌
#verbose 2: 막대 빼고 간결하게
#오래된 데이터라면 verbose0을 두는 것이 좋음


#4. 평가 예측
loss, mae = model.evaluate(x, y, batch_size=1) #3.test

print('loss: ' , loss) # mse 출력
print('mae: ', mae)

x_input = array([[6.5,7.5,8.5], [50,60,70], [70,80,90], [100,110,120]]) # (3,) -> (1, 3) -> (1 , 3, 1) 전체를 곱한 값이 같기 때문에 reshape 가능
x_input = x_input.reshape(4,3,1)

y_predict = model.predict(x_input)
print(y_predict)
