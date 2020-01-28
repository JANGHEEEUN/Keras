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
from keras.models import Sequential, Model #함수형 모델을 쓰기 위해 import Model
from keras.layers import Dense, Input #함수형 모델에는 Input Layer가 추가적으로 들어감

# input1 = Input(shape=(3,))
# dense1 = Dense(5, activation='relu')(input1)
# dense2 = Dense(2)(dense1)
# dense3 = Dense(3)(dense2)
# output1 = Dense(1)(dense3) 

input1 = Input(shape=(3,)) 
x = Dense(5, activation='relu')(input1)
x = Dense(2)(x)
x = Dense(3)(x)
output1 = Dense(1)(x) 

#단을 dense 모델로 늘려주는데 맨 뒤 괄호에 앞 레이어의 이름을 넣어줌

model = Model(inputs = input1, outputs=output1)
#제일 하단에 함수형 모델임을 명시

model.summary()
'''
#3. 훈련- matrics: mse
model.compile(loss='mse', optimizer='Adam', metrics=['mse'])
model.fit(x_train,y_train,epochs=100, batch_size=10, validation_data=(x_val,y_val)) 


#4. 평가 예측
loss, mse = model.evaluate(x_test,y_test, batch_size=1) #3.test
print('mse:' , mse)
print('loss:' , loss)


x_prd = np.array([[501,502,503]])
aaa = model.predict(x_prd, batch_size=1)
print(aaa)

y_predict = model.predict(x_test, batch_size=1)
'''


