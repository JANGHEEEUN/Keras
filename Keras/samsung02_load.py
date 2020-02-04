import numpy as np
import pandas as pd

samsung = np.load('Downloads/data/samsung.npy')
kospi200 = np.load('Downloads/data/kospi.npy')

# print(samsung)
# print(samsung.shape)
# print(kospi200)
# print(kospi200.shape)

def split_xy5(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :]
        tmp_y = dataset[x_end_number:y_end_number,3]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy5(samsung, 5, 1)

# print(x.shape) #(421,5,5)
# print(y.shape) #(421,1)
# print(x[0,:], '\n', y[0])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.3, shuffle=False)
print(x_train.shape)
print(x_test.shape)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])

#데이터 전처리
#StandardScaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)



