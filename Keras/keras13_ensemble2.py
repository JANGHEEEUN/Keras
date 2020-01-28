import numpy as np 

#1. 데이터 
x1 = np.array([range(1,101), range(101,201), range(301,401)])
x2 = np.array([range(1001,1101), range(1101,1201), range(1301,1401)])

y1 = np.array([range(1,101), range(101,201), range(301,401)])
y2 = np.array([range(1001,1101), range(1101,1201), range(1301,1401)])
y3= np.array([range(1,101),range(101,201),range(301,401)])


x1= np.transpose(x1)
x2= np.transpose(x2)
y1= np.transpose(y1)
y2= np.transpose(y2)
y3= np.transpose(y3)

from sklearn.model_selection import train_test_split

x1_train, x1_test, x2_train, x2_test, y1_train, y1_test = train_test_split(x1, x2, y1, test_size=0.4, shuffle=False)
x1_test, x1_val, x2_test, x2_val, y1_test, y1_val = train_test_split(x1_test, x2_test, y1_test, test_size=0.5, shuffle=False)

y2_train, y2_test, y3_train, y3_test = train_test_split(y2, y3, test_size=0.4, shuffle=False)
y2_test, y2_val, y3_test, y3_val = train_test_split(y2_test, y3_test, test_size=0.5, shuffle=False)


print(y3_train.shape) #(60,3)
print(y3_test.shape) #(20,3)
print(y3_val.shape) #(20,3)


#2. 모델 구성
from keras.models import Sequential, Model 
from keras.layers import Dense, Input

input1 = Input(shape=(3,))
dense1 = Dense(5)(input1)
dense2 = Dense(2)(dense1)
dense3 = Dense(3)(dense2)
output1 = Dense(11)(dense3) 

input2 = Input(shape=(3,))
dense21 = Dense(5)(input2)
dense22 = Dense(2)(dense21)
dense23 = Dense(3)(dense22)
output2 = Dense(5)(dense23) #concatenate를 할 경우 output을 y1_train의 col과 맞추지 않아도 됨

from keras.layers.merge import concatenate #concatenate: (모델을) 사슬처럼 엮다
merge1 = concatenate([output1, output2]) 

middle1 = Dense(4)(merge1)
middle2 = Dense(7)(middle1)
middle3 = Dense(1)(middle2) #현재 merge된 마지막 레이어

output_1 = Dense(30)(middle3) #1번째 output 모델
output_1 = Dense(3)(output_1) #Dense(n)을 col과 맞추기

output_2 = Dense(30)(middle3) #2번째 output 모델
output_2 = Dense(5)(output_2)
output_2 = Dense(3)(output_2)

output_3 = Dense(30)(middle3) #3번째 output 모델
output_3= Dense(3)(output_3)

model = Model(inputs = [input1, input2], outputs=[output_1, output_2, output_3])
#model input이 2개이므로 리스트로 넣어줌

model.summary()
#제일 하단에 함수형 모델임을 명시

#3. 훈련- matrics: mse
model.compile(loss='mse', optimizer='Adam', metrics=['mse'])
model.fit([x1_train, x2_train], [y1_train, y2_train, y3_train],epochs=200, batch_size=10, validation_data=([x1_val, x2_val], [y1_val, y2_val, y3_val])) 


#4. 평가 예측
aaa = model.evaluate([x1_test, x2_test],[y1_test, y2_test, y3_test], batch_size=1) #3.test

print('aaa:' , aaa)


x1_prd = np.array([[501,502,503],[504,505,506],[507,508,509]])
x2_prd = np.array([[501,502,503],[504,505,506],[507,508,509]])

x1_prd = np.transpose(x1_prd)
x2_prd = np.transpose(x2_prd)

bbb = model.predict([x1_prd, x2_prd], batch_size=1)
print(bbb)

y1_predict = model.predict([x1_test, x2_test], batch_size=1) #(20,3)이 3개 - list
print(y1_predict[0])


#RMSE 구하기 - 3개를 구해서 평균내는 방법
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
    #np.sqrt: 루트
# print("RMSE : ", RMSE(y1_test, y1_predict))

rmse1 = RMSE(y1_predict[0], y1_test)
rmse2 = RMSE(y1_predict[1], y2_test)
rmse3 = RMSE(y1_predict[2], y3_test)

rmse = (rmse1+rmse2+rmse3)/3.0
print("rmse:  ", rmse)


#R2: 3개 구해서 평균내는 방법
from sklearn.metrics import r2_score
# r2_y_predict = r2_score([y1_test,y2_test,y3_test], y1_predict)
# print("R2: ", r2_y_predict)

r2_y_predict1 = r2_score(y1_test, y1_predict[0])
r2_y_predict2 = r2_score(y2_test, y1_predict[1])
r2_y_predict3 = r2_score(y3_test, y1_predict[2])

r2 = (r2_y_predict1+r2_y_predict2+r2_y_predict3)/3.0
print("r2:   ", r2)