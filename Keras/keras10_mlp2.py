import numpy as np 

#1. 데이터 
x = np.array([range(1,101), range(101,201), range(301,401)])
y = np.array([range(101,201)])
y2 = np.array(range(101,201))

print(x.shape) #(3,100)
print(y.shape) #(1,100)
print(y2.shape) #(100,)

x= np.transpose(x)
y= np.transpose(y)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4,shuffle=False) 
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, shuffle=False)


#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(10, input_shape =(3, ))) #input_dim>=2: 열(col)
model.add(Dense(4))
model.add(Dense(5))
model.add(Dense(1))

model.summary()

#3. 훈련- matrics: mse
model.compile(loss='mae', optimizer='Adam', metrics=['mae'])
model.fit(x_train,y_train,epochs=100, batch_size=60, validation_data=(x_val,y_val)) 
    
#4. 평가 예측
loss, mae = model.evaluate(x_test,y_test, batch_size=1) #3.test
print('mae:' , mae)
print('loss:' , loss)


x_prd = np.array([[501,502,503],[504,505,506], [507,508,509]])
x_prd = np.transpose(x_prd)
aaa = model.predict(x_prd, batch_size=1)
print(aaa)

y_predict = model.predict(x_test, batch_size=20)


#RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
    #np.sqrt: 루트
print("RMSE : ", RMSE(y_test, y_predict))


#R2: 회귀모델 오류 지표 - 평균 제곱 오차
#    0~1 사이값으로 결정됨: 1에 근사할수록 good (RMSE는 낮을수록 good)
#    RSME와 R2가 둘 다 높거나 둘 다 낮은 경우 모델을 다시 짜야 함
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2: ", r2_y_predict)



