import numpy as np

x = np.array(range(1,21))
y = np.array(range(1,21))

x =  x.reshape(20,1)

print(x.shape) #(20,1)
print(y.shape) #(20,)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.5, random_state=66, shuffle = False
)

print("===============================")
print(x_train)
print("===============================")
print(x_test)
print("===============================")

from sklearn.preprocessing import MinMaxScaler, StandardScaler


scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print("===============================")
print(x_train)
print("===============================")
print(x_test)
print("===============================")

#결과를 보면 0~1까지 값으로 정리되지 않음
#이 값과 매치가 되는 y 값과 정확히 매치되고 있음
#0.1 = 1 | 1.1 = 11 | 1.6 = 16
#그래서 정확한 전처리
#만약 범위값 밖으로 되어있더라도 xtrain 하나로만 전처리한 데이터를 transform으로 처리해도 문제 없음
