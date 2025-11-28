import tensorflow as tf
import numpy as np

# 1. 데이터 로드
(trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()

# 2. 전처리 (0~1로 정규화)
trainX = trainX / 255.0
testX = testX / 255.0

# 3. 데이터 형태 변경 (CNN이 아님 → (28,28,1) 필요 없음)
# Functional API 기본 Dense 모델이므로 reshape 불필요
trainX = trainX.reshape((trainX.shape[0], 28,28,1))

# 4. Functional API 모델 정의
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Model

inputs = Input(shape=(28, 28))          # input layer
x = Flatten()(inputs)                   # flatten layer
x = Dense(128, activation='relu')(x)    # hidden1
x = Dense(64, activation='relu')(x)     # hidden2
outputs = Dense(10, activation='softmax')(x)  # output layer (10 classes)

model = Model(inputs=inputs, outputs=outputs)

# 5. 모델 구조 확인
model.summary()

# 6. Compile
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# 7. 학습
model.fit(
    trainX, trainY,
    validation_data=(testX, testY),
    epochs=5
)

# 8. 평가
score = model.evaluate(testX, testY)
print("Test Loss:", score[0])
print("Test Accuracy:", score[1])
