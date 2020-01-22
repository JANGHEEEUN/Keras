import numpy as np 

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# print(x.shape)
# print(y.shape)

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(5, input_dim = 1))
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(1))

# y = wx+b
# w: weight, 가중치 - 그래프의 기울기 값
# b

#만약 ax^2 + bx + c의 형태로 데이터가 분포한다면 미분
# y' = 2ax+b -> 2a를 w로 치환
# y' = wx+b

model.summary()
#(None, 5) - input이 0행 1열, 노드 5개
#param 10 
# w값은 한 번 레이어가 내려갈 때마다 연산됨
# 



'''
#3. 훈련
model.compile(loss='mse', optimizer='Adam', metrics=['mae'])
model.fit(x,y,epochs=100, batch_size=1)
    
#4. 평가 예측
loss, mse = model.evaluate(x,y, batch_size=1)
print('mse:' , mse)
print('loss:' , loss)

x_prd = np.array([11,12,13])
aaa = model.predict(x_prd, batch_size=1)
print(aaa)

bbb = model.predict(x, batch_size=1)
print(bbb)
'''