import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

# ---------- 데이터셋 불러오기 (예: 이미지 64x64 -> 150x150로 리사이즈) ----------
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    './dataset',
    image_size=(150, 150),
    batch_size=32,
    subset='training',
    validation_split=0.2,
    seed=1234
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    './dataset',
    image_size=(150, 150),
    batch_size=32,
    subset='validation',
    validation_split=0.2,
    seed=1234
)

# 전처리 (0~1 스케일링)
train_ds = train_ds.map(lambda x, y: (tf.cast(x/255.0, tf.float32), y))
val_ds = val_ds.map(lambda x, y: (tf.cast(x/255.0, tf.float32), y))

# ---------- InceptionV3 불러오기 ----------
base_model = InceptionV3(
    weights='imagenet',
    include_top=False,
    input_shape=(150,150,3)
)
base_model.trainable = False   # 전이학습 1단계: 기본 가중치는 freeze

# ---------- Custom Layer 쌓기 ----------
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)   # 이진분류 가정

model_inception = Model(inputs=base_model.input, outputs=output)

# ---------- Compile ----------
model_inception.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ---------- Train ----------
history_inception = model_inception.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

model_cnn = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model_cnn.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history_cnn = model_cnn.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

import matplotlib.pyplot as plt

def plot_compare(history1, history2, label1="Model1", label2="Model2"):
    plt.figure(figsize=(12,5))

    # Accuracy 비교
    plt.subplot(1,2,1)
    plt.plot(history1.history['accuracy'])
    plt.plot(history2.history['accuracy'])
    plt.title('Accuracy Comparison')
    plt.legend([label1, label2])

    # Loss 비교
    plt.subplot(1,2,2)
    plt.plot(history1.history['loss'])
    plt.plot(history2.history['loss'])
    plt.title('Loss Comparison')
    plt.legend([label1, label2])

    plt.show()

plot_compare(history_inception, history_cnn, "InceptionV3", "Basic CNN")
