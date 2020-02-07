from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
cancer = load_breast_cancer()

scaler = StandardScaler()
scaler.fit(cancer.data)
X_scaled = scaler.transform(cancer.data)

from sklearn.decomposition import PCA
pca = PCA(n_components=5) #n_components의 값에 따라 축소된 데이터 col이 변경됨
pca.fit(X_scaled)

x_pca = pca.transform(X_scaled)
print("원본 데이터 형태 : ", X_scaled.shape) #(569, 30) - 유방암 데이터의 총 col 30
print("축소된 데이터 형태 : ", x_pca.shape) #(569,2)
# 30개의 col이 2개로 축소됨
# 이 2 col은 원래 컬럼이 아니라 30개의 col이 특성값만 모여 2개로 압축된, 변경된 컬럼

#cifar는 총 col이 약 3000개 이걸 다 돌리면 오래걸림- 이런 경우에 빠른 결과를 얻기 위해 PCA 사용
#하지만 PCA는 조작된 값이라 엄밀한 값에는 틀릴 수 있지만, 머신러닝의 가장 큰 장점인 빠른 피드백을 얻을 수 있음