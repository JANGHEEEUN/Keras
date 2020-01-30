import numpy as np 

#1. 데이터 
x1 = np.array([range(1,101), range(101,201), range(301,401)])
x2 = np.array([range(1001,1101), range(1101,1201), range(1301,1401)])

y1= np.array([range(101,201)])


x1= np.transpose(x1)
x2= np.transpose(x2)
y1= np.transpose(y1)

from sklearn.model_selection import train_test_split

x1_train, x1_test, x2_train, x2_test, y1_train, y1_test = train_test_split(x1, x2, y1, test_size=0.4, shuffle=False)
x1_test, x1_val, x2_test, x2_val, y1_test, y1_val = train_test_split(x1_test, x2_test, y1_test, test_size=0.5, shuffle=False)

print(x1_train.shape) #(50,3)
print(x2_train.shape) #(60,3)
print(y1_train.shape) #(60,1)


#2. 모델 구성
from keras.models import Sequential, Model 
from keras.layers import Dense, Input
input1 = Input(shape=(3,))
dense1 = Dense(5)(input1)
dense2 = Dense(2)(dense1)
dense3 = Dense(3)(dense2)
output1 = Dense(1)(dense3) 

input2 = Input(shape=(3,))
dense21 = Dense(5)(input2)
dense22 = Dense(2)(dense21)
dense23 = Dense(3)(dense22)
output2 = Dense(1)(dense23) #concatenate를 할 경우 output을 y1_train의 col과 맞추지 않아도 됨

# from keras.layers.merge import concatenate #concatenate: (모델을) 사슬처럼 엮다 #concatenate function
# merge1 = concatenate([output1, output2]) 

from keras.layers.merge import Concatenate #Concatenate Class
merge1 = Concatenate()([output1, output2])

middle1 = Dense(4)(merge1)
middle2 = Dense(7)(middle1)
output = Dense(1)(middle2)


model = Model(inputs = [input1, input2], outputs=output)
#model input이 2개이므로 리스트로 넣어줌

model.summary()
#제일 하단에 함수형 모델임을 명시

#3. 훈련- matrics: mse
model.compile(loss='mse', optimizer='Adam', metrics=['mse'])
model.fit([x1_train, x2_train], y1_train,epochs=200, batch_size=10, validation_data=([x1_val, x2_val], y1_val)) 


#4. 평가 예측
loss, mse = model.evaluate([x1_test, x2_test], y1_test, batch_size=1) #3.test
print('mse:' , mse)
print('loss:' , loss)


x1_prd = np.array([[501,502,503],[504,505,506],[507,508,509]])
x2_prd = np.array([[501,502,503],[504,505,506],[507,508,509]])
aaa = model.predict([x1_prd, x2_prd], batch_size=1)
print(aaa)

y_predict = model.predict([x1_test, x2_test], batch_size=1)



#RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
    #np.sqrt: 루트
print("RMSE : ", RMSE(y1_test, y_predict))


#R2: 회귀모델 오류 지표 - 평균 제곱 오차
#    0~1 사이값으로 결정됨: 1에 근사할수록 good (RMSE는 낮을수록 good)
#    RSME와 R2가 둘 다 높거나 둘 다 낮은 경우 모델을 다시 짜야 함
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y1_test, y_predict)
print("R2: ", r2_y_predict)

