#linear svc. kNeighbor Classifier
#iris - 3진 분류

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

iris_data = pd.read_csv("./Keras/data/iris.csv", encoding='utf-8',
                        names=['a','b','c','d','y'])

y = iris_data.loc[:,"y"]
x = iris_data.loc[:, ["a","b","c","d"]]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,train_size=0.7, shuffle=True)

clf = SVC()
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
print("정답률: ", accuracy_score(y_test, y_pred))
