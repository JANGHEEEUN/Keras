import numpy as np 

#모델을 저장했으므로 1.데이터 삭제

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(10, activation = 'relu', input_shape=(3,1))) # input_shape(열, 몇 개씩 자르는지) #(3,1): 열이 3개고 데이터 셋을 1개씩 잘라서 작업
model.add(Dense(5))                                         # 1개씩 자르면 결과는 잘 나옴 but 느림 <-> 2개씩 자르면 빠르지만 결과에 영향 << 하이퍼 파라미터 수정
model.add(Dense(1))


#이미 훈련된 모델을 불러오므로 훈련 이하 삭제

model.save('./save/savetest01.h5')
print('저장 잘 됐다.')