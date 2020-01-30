#gru나 simple rnn도 lstm과 사용법은 같음

from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

x = array([[1,2,3], [2,3,4],[3,4,5], [4,5,6], [5,6,7]]) #(5,3)
y = array([4,5,6,7,8]) # (5,) - 벡터

x = x.reshape(x.shape[0], x.shape[1], 1) # x를 (5,3,1)로 reshape - 뒤에 몇 개씩 자르는지 붙여줘야 함
                                         # 5,3,1은 곱했을 때 원 데이터와 같이 15가 나옴

model = Sequential()
model.add(LSTM(10, activation = 'relu', input_shape=(3,1))) # input_shape(열, 몇 개씩 자르는지) #(3,1): 열이 3개고 데이터 셋을 1개씩 잘라서 작업
model.add(Dense(5))                                         # 1개씩 자르면 결과는 잘 나옴 but 느림 <-> 2개씩 자르면 빠르지만 결과에 영향 << 하이퍼 파라미터 수정
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='Adam', metrics=['mae']) # metrics mae = 반환값 2개
model.fit(x, y, epochs=300, batch_size=1) 


#4. 평가 예측
loss, mae = model.evaluate(x, y, batch_size=1) #3.test

print('loss: ' , loss) # mse 출력
print('mae: ', mae)

x_input = array([6,7,8]) # (3,) -> (1, 3) -> (1 , 3, 1) 전체를 곱한 값이 같기 때문에 reshape 가능
x_input = x_input.reshape(1,3,1)

y_predict = model.predict(x_input)
print(y_predict)
