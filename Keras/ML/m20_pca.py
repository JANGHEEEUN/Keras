from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
cancer = load_breast_cancer()

scaler = StandardScaler()
scaler.fit(cancer.data)
X_scaled = scaler.transform(cancer.data)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X_scaled)

x_pca = pca.transform(X_scaled)
print("원본 데이터 형태 : ", X_scaled.shape) #(569, 30) - 유방암 데이터의 총 col 30
print("축소된 데이터 형태 : ", x_pca.shape) #(569,2)
