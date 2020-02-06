import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

wine = pd.read_csv("./Keras/data/winequality-white.csv",  delimiter=";",encoding="utf-8")
y = wine["quality"]
x = wine.drop("quality", axis=1)

newlist = []
for v in list(y):
    if v<=4:
        newlist += [0]
    elif v<=7:
        newlist += [1]
    else:
        newlist += [2]

y = newlist

# one hot encoding
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
y = np_utils.to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, test_size=0.2 ,train_size=0.8, shuffle = True
)

model = Sequential()
model.add(Dense(12,input_dim=11,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=500,batch_size=10)

loss, accuracy = model.evaluate(x_test, y_test)
print(accuracy)
# predict value softmax > one hot encoding
y_pred = model.predict_classes(x_test)
# y_pred = np.argmax(y_pred,axis=1)
# y_pred = encoder.inverse_transform(y_pred)
# y_pred = encoder.transform(y_pred)
# y_pred = np_utils.to_categorical(y_pred)
print(y_pred)

# print('정답률: ', accuracy_score(y_test,y_pred))