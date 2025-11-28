import tensorflow as tf
import numpy as np

# 1) 데이터 로드
(trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()

# 2) 0~255 → 0~1 스케일링
trainX = trainX / 255.0
testX = testX / 255.0

# 3) CNN 입력 형태(28,28,1)로 변환
trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))

# 4) 모델 정의 (Convolution + Pooling 포함)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 5) 모델 컴파일
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# 6) 학습 (overfitting 방지: validation_data 사용)
history = model.fit(
    trainX, trainY,
    validation_data=(testX, testY),
    epochs=10
)

# 7) 테스트 데이터 평가
score = model.evaluate(testX, testY)
print("테스트 Loss:", score[0])
print("테스트 Accuracy:", score[1])
