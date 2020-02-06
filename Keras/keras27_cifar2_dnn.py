from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.callbacks import EarlyStopping
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data();

print(x_train)
print(y_train)

print(x_train.shape) # (50000, 32, 32, 3)
print(x_test.shape)  #(10000, 32, 32, 3)
 
print(y_train.shape) #(50000, 1)
print(y_test.shape) #(10000,1)


x_train = x_train.reshape(x_train.shape[0], 32*32*3).astype('float32')/255
x_test = x_test.reshape(x_test.shape[0], 32*32*3).astype('float32')/255

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(y_train.shape) 

model = Sequential()
model.add(Dense(256, input_shape=(32*32*3, )))
model.add(Dense(84))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])


early_stopping = EarlyStopping(monitor='loss', patience=20)

model.fit(x_train, y_train, validation_split=0.2, epochs=200, batch_size=300, verbose=2,
          callbacks=[early_stopping])

acc = model.evaluate(x_test, y_test)

print(acc)

