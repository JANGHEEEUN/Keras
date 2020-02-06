import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
from sklearn.utils.testing import all_estimators


warnings.filterwarnings('ignore')

iris_data = pd.read_csv("./Keras/data/iris2.csv", encoding="utf-8")

y = iris_data.loc[:, "Name"]
x = iris_data.loc[:, ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]

warnings.filterwarnings('ignore')
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, train_size=0.8, shuffle=True)

warnings.filterwarnings('ignore')
allAlgorithms = all_estimators(type_filter="classifier")

print(allAlgorithms)
print(len(allAlgorithms))
print(type(allAlgorithms))

for(name, algorithm) in allAlgorithms:
    # 각 알고리즘 객체 생성하기
    clf = algorithm()
    
    # 학습하고 평가하기
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(name, "의 정답률; ", accuracy_score(y_test, y_pred))