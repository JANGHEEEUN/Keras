import numpy as np 

#1. 데이터 
x = np.array([range(1,101), range(101,201), range(301,401)])
y = np.array([range(101,201)])

# x= np.transpose(x)
x = x.reshape(x.shape[1], x.shape[0], 1)
y= np.transpose(y)
print(x.shape) #(3,100)
print(y.shape) #(1,100)

# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4,shuffle=False) 
# x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, shuffle=False)


#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense, LSTM
# model = Sequential()

# model.add(Dense(10, input_shape =(3, ))) #input_dim>=2: 열(col)
# model.add(Dense(4))
# model.add(Dense(5))
# model.add(Dense(1))

model = Sequential()
model.add(LSTM(10, activation = 'relu', input_shape=(3,1))) # input_shape(열, 몇 개씩 자르는지) #(3,1): 열이 3개고 데이터 셋을 1개씩 잘라서 작업
model.add(Dense(5))                                         # 1개씩 자르면 결과는 잘 나옴 but 느림 <-> 2개씩 자르면 빠르지만 결과에 영향 << 하이퍼 파라미터 수정
model.add(Dense(1))

model.summary()

#early stopping
from keras.callbacks import EarlyStopping, TensorBoard
tb_hist = TensorBoard(log_dir='./graph',
                      histogram_freq=0,
                      write_graph=True,
                      write_images=True )

#early stopping
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto') 

#3. 훈련- matrics: mse
model.compile(loss='mse', optimizer='Adam', metrics=['mae'])
model.fit(x, y, epochs=100, batch_size=1, verbose=1, callbacks=[early_stopping, tb_hist]) 
# model.fit(x_train,y_train,epochs=100, batch_size=60, validation_data=(x_val,y_val), callbacks=[early_stopping, tb_hist]) 
    
#4. 평가 예측
loss, mae = model.evaluate(x, y, batch_size=1) #3.test
print('mae:' , mae)
print('loss:' , loss)


x_input = np.array([606,607,608]) # (3,) -> (1, 3) -> (1 , 3, 1) 전체를 곱한 값이 같기 때문에 reshape 가능
x_input = x_input.reshape(1,3,1)

y_predict = model.predict(x_input)
print(y_predict)

# x_prd = np.array([[501,502,503],[504,505,506], [507,508,509]])
# x_prd = np.transpose(x_prd)
# aaa = model.predict(x_prd, batch_size=1)
# print(aaa)

# y_predict = model.predict(x_test, batch_size=20)


# #RMSE 구하기
# from sklearn.metrics import mean_squared_error
# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))
#     #np.sqrt: 루트
# print("RMSE : ", RMSE(y_test, y_predict))


# #R2: 회귀모델 오류 지표 - 평균 제곱 오차
# #    0~1 사이값으로 결정됨: 1에 근사할수록 good (RMSE는 낮을수록 good)
# #    RSME와 R2가 둘 다 높거나 둘 다 낮은 경우 모델을 다시 짜야 함
# from sklearn.metrics import r2_score
# r2_y_predict = r2_score(y_test, y_predict)
# print("R2: ", r2_y_predict)



