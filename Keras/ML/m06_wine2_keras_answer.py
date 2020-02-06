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

iris_data = pd.read_csv('csv/iris.csv',encoding='utf-8',names=['a','b','c','d','y'])
y = iris_data.loc[:,'y']
x = iris_data.loc[:,['a','b','c','d']]

# one hot encoding
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
y = np_utils.to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, test_size=0.2 ,train_size=0.7, shuffle = True
)

model = Sequential()
model.add(Dense(12,input_dim=4,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(3,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=200,batch_size=10)

# predict value softmax > one hot encoding
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred,axis=1)
y_pred = np_utils.to_categorical(y_pred)
print(y_pred)

print('정답률: ', accuracy_score(y_test,y_pred))