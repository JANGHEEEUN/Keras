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

model.add(Dense(10, input_dim = 1))
model.add(Dense(5))
model.add(Dense(1))


#3. 훈련
model.compile(loss='mse', optimizer='Adam', metrics=['mae'])
    #loss: 손실 - 낮을수록 좋음 
    #mse: mean_squared_error - 낮을수록 좋음
    #optimizer: 최적화 - 보통 adam 사용
    # metrics: 실행결과를 acc로 보여줌[틀림] - 회귀모델의 지표는 
# model.fit(x,y,epochs=100, batch_size=1)
model.fit(x,y,epochs=100, batch_size=5)
    #epoch: 학습반복횟수 
    #batch가 100인 것과 1인 것 중 뭐가 좋은지는 결과를 봐야 알 수 있음
    #batch_size: 전체 데이터를 자르는 단위 - 낮을 때 정확도가 높아짐
    #            경험상으로 넣을 수 있음
    #batch_size를 지웠는데도 동작 >> default가 있음: 32
    
#4. 평가 예측
loss, mse = model.evaluate(x,y, batch_size=5)
print('mse:' , mse)
print('loss:' , loss)

x_prd = np.array([11,12,13])
aaa = model.predict(x_prd, batch_size=5)
print(aaa)

bbb = model.predict(x, batch_size=5)
print(bbb)

#회귀모델에서는 mse, mae를 사용
#mae - mean absolute error: 절댓값
#mse와 mae: 둘 다 양수값을 가짐 
#rmae, rmse: mae, mse의 값에 루트 씌워준 것
#하지만 rmae, rmse는 compile할 때 적용하진 않음
#이 4가지는 수치가 낮을수록 좋음
