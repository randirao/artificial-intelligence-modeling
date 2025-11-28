import tensorflow as tf
import pandas as pd

data = pd.read_csv('gpascore.csv')

# 결측치 제거
data = data.dropna()

# y 데이터 (정답)
y데이터 = data['admit'].values
print(y데이터)

# x 데이터 (gre, gpa, rank 추출)
x데이터 = []
for i, rows in data.iterrows():
    x데이터.append([rows['gre'], rows['gpa'], rows['rank']])

print(x데이터)
