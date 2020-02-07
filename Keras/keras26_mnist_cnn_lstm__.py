from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, LSTM
from keras.callbacks import EarlyStopping
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train)
# print(y_train)

print(x_train.shape) #(60000,28,28) 하지만 행이 무시되므로 28,28인 2차원 -> 3차원으로 reshape 필요
print(y_train.shape)

#이렇게 한 줄에 전처리 가능한 경우는 mnist처럼 완벽한 데이터인 경우에만 가능
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(x_test.shape[0],28,28,1).astype('float32')/255

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train) #onehot encoding하기 위해
y_test = np_utils.to_categorical(y_test)

print(y_train.shape) # (60000, 10) # onehot encoding으로 10개의 값으로 바뀜 -> dense output = 10
                    #마지막 layer activation = softmax 
                    #onehot -> 1: 1000000000
                            #  2: 0100000000
                            #  3: 0010000000

from keras.layers import Reshape

model = Sequential()
model.add(Conv2D(16, (2,2), strides=(2,2), padding='same',
                 activation = 'relu',
                input_shape=(28, 28, 1)))
# model.add(MaxPooling2D(2,2))
# model.add(Conv2D(16, (2,2), activation='relu', padding='same'))
model.add(Reshape((14*4,14*4)))
model.add(LSTM(4, activation='relu', input_shape=(14*4,14*4)))
model.add(Dense(10, activation='softmax'))

model.summary()


# model = Sequential();
# model.add(Conv2D(16, (2,2), input_shape = (28,28,1)))
# model.add(Reshape((-1,4)))
# model.add(LSTM(256,input_shape=(28*7,4)))
# model.add(Dense(16))
# model.add(Dense(10, activation='softmax')) #다중분류는 무조건 softmax
# model.summary()


model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
#다중분류에 accuracy 사용가능한 이유
#이전에는 답을 확신할 수 없었음. 그렇기에 정확도를 낼 수 없음
#하지만 지금은 무조건 y값이 10개 밖에 없음 그래서 원래 y값ㅇ ㅣ5인데 pred가 4이면 틀린 것, 5면 맞는 것 -> 명확하기에 accuracy 사용 가능


early_stopping = EarlyStopping(monitor='loss', patience=20)

model.fit(x_train, y_train, validation_split=0.2, epochs=2, batch_size=100, verbose=2,
          callbacks=[early_stopping])

acc = model.evaluate(x_test, y_test)

print(acc)
