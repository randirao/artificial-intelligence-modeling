import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('gpascore.csv')
data = data.dropna()
x = []

y = data['admit'].values
print(y)

# 과제 작성 부분
for index, row in data.iterrows():
    x.append([row['gre'], row['gpa'], row['rank']])
print(x)

# 데이터 정규화
scaler = StandardScaler()
x = scaler.fit_transform(x)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(64, activation='swish'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy' , metrics=['accuracy'])

model.fit( x, np.array(y), epochs=200)
