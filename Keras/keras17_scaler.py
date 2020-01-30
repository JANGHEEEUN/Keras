from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

x = array([[1,2,3], [2,3,4],[3,4,5], [4,5,6], 
           [5,6,7], [6,7,8], [7,8,9], [8,9,10],
           [9,10,11], [10,11,12], [20000,30000,40000], 
           [30000,40000,50000], [40000,50000,60000], [100,200,300]]) #(13,3)
y = array([4,5,6,7,8,9,10,11,12,13,50000,60000,70000,400]) # (13,) - 벡터

from sklearn.preprocessing import MinMaxScaler

