import numpy as np
import pandas as pd

df1 = pd.read_csv('Downloads/samsung.csv', index_col=0, header=0, encoding='cp949', sep=',') 
#index_col 0 : 날짜를 index로

# print(df1)
# print(df1.shape)

df2 = pd.read_csv('Downloads/kospi200.csv', index_col=0, header=0, encoding='cp949', sep=',') 


# print(df2.shape)

#kospi200의 모든 데이터
for i in range(len(df2.index)):
    df2.iloc[i,4] = int(df2.iloc[i,4].replace(',',''))

# 삼성전자의 모든 데이터 
for i in range(len(df1.index)):
    for j in range(len(df1.iloc[i])):
        df1.iloc[i,j] = int(df1.iloc[i,j].replace(',',''))

df1 = df1.sort_values(['일자'], ascending = [True])
df2 = df2.sort_values(['일자'], ascending = [True])
# print(df2)

#pandas -> numpy로 해줘야 계산 속도가 빠름

df1 = df1.values #pandas -> np
df2 = df2.values #pandas -> np

print(type(df1), type(df2))
print(df1.shape, df2.shape)

np.save('Downloads/data/samsung.npy', arr = df1)
np.save('Downloads/data/kospi.npy', arr = df2)