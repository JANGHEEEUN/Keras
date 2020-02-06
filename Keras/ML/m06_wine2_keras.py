#linear svc. kNeighbor Classifier
#iris - 3진 분류

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf

wine = pd.read_csv("./Keras/data/winequality-white.csv",  delimiter=";",encoding="utf-8")

y = wine["quality"]
x = wine.drop("quality", axis=1)
print(x.info()) 


x = np.array(x)
y = np.array(y) 

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, shuffle=True)


#3918,11


from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# print(y_train[0:100])

#m06_wine3.py에서 와인 분류가 10개가 아닌 7개인 것을 확인
model = Sequential()
model.add(Dense(60, input_shape = (11,)))
model.add(Dense(52))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100, batch_size = 10, verbose=2)

acc = model.predict(x_test)

loss, accuracy = model.evaluate(x_test, y_test)

print(accuracy)
