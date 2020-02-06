from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

model = Sequential();
model.add(Conv2D(7, (2,2), padding='valid', strides = (2,2), input_shape = (5,5,1)))
model.add(Conv2D(100, (2,2), padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(1))

model.summary()