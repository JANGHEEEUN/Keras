# m10 gridsearch + m12 pipeline

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import warnings
from sklearn.utils.testing import all_estimators
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import numpy as np

warnings.filterwarnings('ignore')

iris_data = pd.read_csv("./Keras/data/iris2.csv", encoding="utf-8")

y = iris_data.loc[:, "Name"]
x = iris_data.loc[:, ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]

warnings.filterwarnings('ignore')
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, train_size=0.8, shuffle=True)


#그리드 서치에서 사용할 매개 변수
parameter = [
    {"svc__C": [1,10,100,1000], "svc__kernel": ["linear"]},
    {"svc__C": [1,10,100,1000], "svc__kernel": ["rbf"], "svc__gamma":[0.001, 0.0001]},
    {"svc__C": [1,10,100,1000], "svc__kernel": ["sigmoid"], "svc__gamma":[0.001, 0.0001]}
]

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

scaler_tuple = ('scaler', MinMaxScaler())

model_tuple = ('svc', SVC())


#pipeline -> gridsearch : 이렇게 해야 최적 모델의 매개변수 확인 가능
pipe = Pipeline([scaler_tuple, model_tuple])

# #그리드 서치
kfold_cv = KFold(n_splits=5, shuffle=True) #5번
model =GridSearchCV(estimator=pipe, param_grid = parameter, cv=kfold_cv) #20번
#svc: 분류모델


model.fit(x_train, y_train)
print("최적의 매개 변수 - ", model.best_estimator_)
 
y_pred = model.predict(x_test)
print("최종 정답률: ", accuracy_score(y_test, y_pred))