from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import classification_report
import numpy as np

boston = load_boston()

# print(boston)
x = boston.data
y = boston.target


x = np.array(x)
y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, shuffle=True)





from sklearn.linear_model import LinearRegression, Ridge, Lasso



model = LinearRegression()
model.fit(x_train,y_train)

y_prd = model.predict(x_test)
print(y_prd)
print("score: ", model.score(x_test, y_test))
# print("정답률: ", accuracy_score(y_test, y_prd))


# model = Ridge()
# model.fit(x,y)
# y_prd = model.predict(x)
# print(y_prd)

# model = Lasso()
# model.fit(x,y)
# y_prd = model.predict(x)
