import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
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
parameter = [
    {"n_estimators": [1,2,3,10,100,200,1000,2000], "criterion": ["entropy"], "max_depth":[2, 3,4, 6]},
    {"n_estimators": [1,2,3,10,100,200,1000,2000], "criterion": ["gini"], "max_depth":[2, 3, 4,6]}
]

#그리드 서치
kfold_cv = KFold(n_splits=5, shuffle=True) #5번
model =GridSearchCV(RandomForestClassifier(), parameter, cv=kfold_cv) #20번
#svc: 분류모델

#총 돌린 횟수: 100번

model.fit(x_train, y_train)
print("최적의 매개 변수 - ", model.best_estimator_)

y_pred = model.predict(x_test)
print("최종 정답률: ", accuracy_score(y_test, y_pred))

#잘 나온 파라미터를 적어두고 그냥 Randomforest에 넣고 돌려봐야 함