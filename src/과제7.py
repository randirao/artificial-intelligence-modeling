import tensorflow as tf
import numpy as np

# 데이터 불러오기
(trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()

# 0~1 스케일링
trainX = trainX / 255.0
testX = testX / 255.0

# 모델 설계
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])

# 컴파일
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0008),
    metrics=['accuracy']
)

# 학습
model.fit(trainX, trainY, validation_data=(testX, testY), epochs=15)

# 평가
score = model.evaluate(testX, testY)
print("test loss:", score[0])
print("test accuracy:", score[1])
