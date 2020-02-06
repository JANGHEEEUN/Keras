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
from sklearn.preprocessing import LabelEncoder

iris_data = pd.read_csv("./Keras/data/iris.csv", encoding='utf-8',
                        names=['a','b','c','d','y'])

y = iris_data.loc[:,"y"]
x = iris_data.loc[:, ["a","b","c","d"]]

y = y.replace('Iris-setosa', '0')
y = y.replace('Iris-versicolor', '1')
y = y.replace('Iris-virginica', '2')

enc = LabelEncoder()
y=enc.fit(y)
y = enc.transform(y)

x = np.array(x)
y = np.array(y)


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,train_size=0.7, shuffle=True)



print(x_train.shape)


from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


model = Sequential()
model.add(Dense(10, input_shape = (4,)))
model.add(Dense(3))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['mae'])

model.fit(x_train, y_train, epochs=200, batch_size = 10, verbose=2)

acc = model.evaluate(x_test, y_test)
# acc = np.argmax(acc)
print(acc)