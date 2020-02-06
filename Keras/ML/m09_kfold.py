import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
import warnings
from sklearn.utils.testing import all_estimators

warnings.filterwarnings('ignore')

iris_data = pd.read_csv("./Keras/data/iris2.csv", encoding="utf-8")

y = iris_data.loc[:, "Name"]
x = iris_data.loc[:, ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]

warnings.filterwarnings('ignore')
allAlgorithms = all_estimators(type_filter="classifier")

kfold_cv = KFold(n_splits=3, shuffle=True)

for(name, algorithm) in allAlgorithms:
    # 각 알고리즘 객체 생성하기
    model = algorithm()
    
    if hasattr(model, "score"): #score가 있는 모델만 사용하겠다.
        score = cross_val_score(model, x, y, cv=kfold_cv) #cross_val_score에 fit이 포함되어 있음
        print(name, "의 정답률: ")
        print(score)
        score_sum = 0
        for i in score:
            score_sum += i
        print(score_sum / len(score))

