import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# GPU 메모리 증가 허용 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 데이터셋 로드
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    './dataset',
    image_size=(128, 128),  # 이미지 크기 증가로 더 나은 특징 학습
    batch_size=32,
    subset="training",
    validation_split=0.2,
    seed=1234
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    './dataset',
    image_size=(128, 128),
    batch_size=32,
    subset="validation",
    validation_split=0.2,
    seed=1234
)

# 클래스 이름 확인
class_names = train_ds.class_names

# 데이터 증강 레이어 (overfitting 방지)
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),  # 좌우 반전
    layers.RandomRotation(0.1),  # 10% 회전
    layers.RandomZoom(0.1),  # 10% 줌
    layers.RandomContrast(0.1),  # 대비 조정
])

# 데이터 정규화 및 최적화
normalization_layer = layers.Rescaling(1./255)

# 성능 최적화를 위한 캐싱 및 프리페칭
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 모델 구축 (overfitting 방지를 위한 설계)
model = keras.Sequential([
    # 입력 레이어 및 정규화
    layers.Input(shape=(128, 128, 3)),
    normalization_layer,
    data_augmentation,

    # 첫 번째 Conv Block
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2),
    layers.Dropout(0.25),

    # 두 번째 Conv Block
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2),
    layers.Dropout(0.25),

    # 세 번째 Conv Block
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2),
    layers.Dropout(0.3),

    # 네 번째 Conv Block
    layers.Conv2D(256, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2),
    layers.Dropout(0.3),

    # Fully Connected Layers
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    # 출력 레이어
    layers.Dense(len(class_names), activation='softmax')
])

# 모델 컴파일
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 콜백 설정 (overfitting 방지)
callbacks = [
    # Early Stopping: validation loss가 개선되지 않으면 학습 중단
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=0
    ),
    # Learning Rate 감소: validation loss가 개선되지 않으면 학습률 감소
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=0
    ),
]

# 모델 학습
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    callbacks=callbacks,
    verbose=1
)

# 모델 저장
model.save('cat_dog_classifier.h5')

# 학습 결과 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy 그래프
axes[0].plot(history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
axes[0].plot(history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].legend(loc='lower right', fontsize=10)
axes[0].grid(True, alpha=0.3)

# Loss 그래프
axes[1].plot(history.history['loss'], 'b-', label='Training Loss', linewidth=2)
axes[1].plot(history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].legend(loc='upper right', fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
plt.show()