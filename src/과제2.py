import tensorflow as tf

# 행렬 A
A = tf.constant([
    [1, 2, 3],
    [4, 5, 6]
], dtype=tf.float32)

# 행렬 B
B = tf.constant([
    [7, 8],
    [9, 10],
    [11, 12]
], dtype=tf.float32)

# 행렬 곱 (A × B)
C = tf.matmul(A, B)
print(C)
