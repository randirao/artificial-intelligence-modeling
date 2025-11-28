import tensorflow as tf

# 데이터
키 = [150, 160, 170, 180]
신발 = [152, 162, 172, 182]

# 학습할 변수 (a, b)
a = tf.Variable(0.1)
b = tf.Variable(0.5)

# 옵티마이저 (무조건 사용해야 함)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# 1000번 학습
for step in range(1000):

    with tf.GradientTape() as tape:
        # 예측값
        예측신발 = a * 키 + b
        # MSE 손실
        loss = tf.reduce_mean((예측신발 - 신발) ** 2)

    # 기울기 계산
    gradients = tape.gradient(loss, [a, b])

    # a, b 업데이트
    optimizer.apply_gradients(zip(gradients, [a, b]))

    # 100번마다 출력
    if step % 100 == 0:
        print(step, "loss:", loss.numpy(), "a:", a.numpy(), "b:", b.numpy())

print("\n=== 최종 결과 ===")
print("a =", a.numpy())
print("b =", b.numpy())
