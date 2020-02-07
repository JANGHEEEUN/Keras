import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import warnings
from sklearn.utils.testing import all_estimators
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

iris_data = pd.read_csv("./Keras/data/iris2.csv", encoding="utf-8")

y = iris_data.loc[:, "Name"]
x = iris_data.loc[:, ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]

warnings.filterwarnings('ignore')
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, train_size=0.6, shuffle=True)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

#한방에 전처리 진행 - 모델과 전처리를 같이
pipe = Pipeline( [("scaler", MinMaxScaler()), ('svm', SVC())] ) # #전처리는 minmaxscaler로 하고 이름을 scaler로 지정
#전처리 모델은 svc

#바로 훈련 진행
pipe.fit(x_train, y_train)
#pipeline이란 모델은 preprocessing이 포함된 모델이다.
#이렇게 생각하면 pipe.score가 되는 것이 이해 가능
#머신러닝 교과서 220p

print("테스트 점수: ", pipe.score(x_test, y_test)) #evaluate와 동일
