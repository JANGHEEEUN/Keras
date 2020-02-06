import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import classification_report

wine = pd.read_csv("./Keras/data/winequality-white.csv",  delimiter=";",encoding="utf-8")

y = wine["quality"]
x = wine.drop("quality", axis=1)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, shuffle=True)

model = RandomForestClassifier()

model.fit(x_train, y_train)

# 평가 예측 - loss를 뺀 acc만 출력
aaa = model.score(x_test, y_test)
print("aaa:" , aaa)

y_prd = model.predict(x_test)
print("정답률: ", accuracy_score(y_test, y_prd))

print(classification_report(y_test, y_prd))