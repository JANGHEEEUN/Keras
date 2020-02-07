from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# decision tree upgrade version -- randomforestClassifier
# 의미있다고 생각되는 col만 모아서 러닝해도 됨 but accuracy 보장X
# 정확도 판단은 accuracy로 판단 
cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
                                                    stratify = cancer.target,
                                                    random_state = 42)

# tree = DecisionTreeClassifier(random_state=0)
# tree = RandomForestClassifier(random_state=0)
tree = GradientBoostingClassifier(random_state=0)

tree.fit(x_train, y_train)
print("훈련 세트 정확도: {:.3f}".format(tree.score(x_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(tree.score(x_test, y_test)))

print("특성 중요도:\n", tree.feature_importances_)

import matplotlib.pyplot as plt
import numpy as np
def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
             align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("특성 중요도")
    plt.ylabel("특성")
    plt.ylim(-1, n_features)
    
plot_feature_importances_cancer(tree)
plt.show()

#decision tree가 100프로 신뢰 가능한가?
#의미있는 피쳐들이 중요함
