import os
import shutil
import tensorflow as tf
import matplotlib.pyplot as plt

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    './dataset',
    image_size=(64, 64),
    batch_size=64,
    subset="training",
    validation_split=0.2,
    seed=1234
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    './dataset',
    image_size=(64, 64),
    batch_size=64,
    subset="validation",
    validation_split=0.2,
    seed=1234
)

def preprocess(image, label):
    image = tf.cast(image / 255.0, tf.float32)
    return image, label

train_ds = train_ds.map(preprocess)
val_ds = val_ds.map(preprocess)

model = tf.keras.models.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal', input_shape=(64,64,3)),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),

    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

print("학습 완료")
