from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM
import numpy as np
# 1. 데이터
x_train = [[0,0], [1,0], [0,1], [1,1]]
y_train = [0,1,1,0]

x_train = np.array(x_train)
y_train = np.array(y_train)


# 2. 모델
model = Sequential()
model.add(Dense(10, input_shape =(2, )))
model.add(Dense(4))
model.add(Dense(5))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['mae'])
# 3. 훈련
model.fit(x_train,y_train, epochs=100, batch_size=10)

# 4. 평가 예측
x_test = [[0,0],[1,0],[0,1],[1,1]]
x_test = np.array(x_test)
y_predict = model.predict(x_test)

print(y_predict)
