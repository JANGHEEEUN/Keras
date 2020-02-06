import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import warnings
from sklearn.utils.testing import all_estimators
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier


warnings.filterwarnings('ignore')

iris_data = pd.read_csv("./Keras/data/iris2.csv", encoding="utf-8")

y = iris_data.loc[:, "Name"]
x = iris_data.loc[:, ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]

warnings.filterwarnings('ignore')
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, train_size=0.8, shuffle=True)

#그리드 서치에서 사용할 매개 변수
parameter = {"n_estimators": [1,10,20,30,40,50,60,70,80,90,100,1000],
              "max_depth": [4, 8, 12, 16],
              "min_samples_split": [3, 5, 7, 9]}

kfold_cv = KFold(n_splits=5, shuffle=True)
model =RandomizedSearchCV(RandomForestClassifier(), parameter, cv=kfold_cv)
model.fit(x_train, y_train)
print("최적의 매개 변수 - ", model.best_estimator_)

y_pred = model.predict(x_test)
print("최종 정답률: ", accuracy_score(y_test, y_pred))