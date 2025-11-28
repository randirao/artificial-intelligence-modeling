import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
import numpy as np

# CIFAR-10 데이터셋 로드
(trainX, trainY), (testX, testY) = cifar10.load_data()

# 데이터 정규화
trainX = trainX / 255.0
testX = testX / 255.0

# 입력: CIFAR-10은 32x32x3 컬러 이미지
inputs = layers.Input(shape=(32, 32, 3))

# 첫 번째 브랜치: Conv → MaxPool 경로
branch1 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
branch1 = layers.MaxPooling2D((2, 2))(branch1)  # (None, 16, 16, 32)

# 두 번째 브랜치: Conv → MaxPool 경로 (다른 필터 크기)
branch2 = layers.Conv2D(32, (5, 5), padding='same', activation='relu')(inputs)
branch2 = layers.MaxPooling2D((2, 2))(branch2)  # (None, 16, 16, 32)

# 두 브랜치를 Concatenate (채널 방향으로 합침)
concat = layers.Concatenate(axis=-1)([branch1, branch2])  # (None, 16, 16, 64)

# Concatenate 결과 처리
x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(concat)
x = layers.MaxPooling2D((2, 2))(x)  # (None, 8, 8, 64)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)

# 출력층 (CIFAR-10은 10개 클래스)
outputs = layers.Dense(10, activation='softmax')(x)

# 모델 생성
model = models.Model(inputs=inputs, outputs=outputs)

# 모델 컴파일
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 모델 요약 출력
model.summary()

# 모델 구조 시각화
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

# 모델 학습
print("\n학습 시작...")
history = model.fit(
    trainX, trainY,
    validation_data=(testX, testY),
    epochs=5,
    batch_size=64
)
