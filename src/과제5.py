import tensorflow as tf
import pandas as pd
import numpy as np

# 데이터 로드 및 전처리
data = pd.read_csv('gpascore.csv')
data = data.dropna()

y데이터 = data['admit'].values

x데이터 = []
for i, rows in data.iterrows():
    x데이터.append([rows['gre'], rows['gpa'], rows['rank']])

x데이터 = np.array(x데이터)
y데이터 = np.array(y데이터)

# 모델 구성
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),  # 노드 증가 + relu
    tf.keras.layers.Dense(256, activation='relu'),  # 깊은 모델
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # 확률 예측
])

# 모델 컴파일
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),   # 학습률 조정
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 학습
model.fit(x데이터, y데이터, epochs=3000)  # epochs 증가
