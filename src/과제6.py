import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

tf.keras.datasets.fashion_mnist.load_data()

# Fashion-MNIST 데이터 불러오기 (옷 사진 데이터)
(trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()

# 28x28 사진을 784개 숫자로 펼치고, 0~255 값을 0~1로 줄임
trainX = trainX.reshape(trainX.shape[0], -1) / 255.0
testX = testX.reshape(testX.shape[0], -1) / 255.0
# print(trainX)
# print(trainY)
# print(trainX[0])
# print(trainX.shape)
# print(testX.shape)
# print(trainY)
# print(trainY)

# plt.imshow(trainX[0])
# plt.gray()
# plt.colorbar()
# plt.show()

# data = pd.read_csv('gpascore.csv')
# data = data.dropna()
# x = []
#
# y = data['admit'].values
# print(y)
#
# # 과제 작성 부분
# for index, row in data.iterrows():
#     x.append([row['gre'], row['gpa'], row['rank']])
# print(x)
#
# # 데이터 정규화
# scaler = StandardScaler()
# x = scaler.fit_transform(x)

# 신경망 모델 만들기
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='tanh', input_shape=(784,)),  # 첫 번째 층: 784개 입력받아 128개로
    tf.keras.layers.Dense(128, activation='tanh'),                      # 두 번째 층: 128개
    tf.keras.layers.Dense(64, activation='swish'),                      # 세 번째 층: 64개
    tf.keras.layers.Dense(10, activation='softmax')                     # 마지막 층: 10가지 옷 종류로 분류
])

# 모델 설정: 학습 방법, 손실함수, 정확도 측정
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 학습: 훈련 데이터로 학습하고 테스트 데이터로 검증, 200번 반복
model.fit(trainX, trainY, validation_data=(testX, testY), epochs=200)
