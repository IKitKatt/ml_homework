import numpy as np
import time

# Генерация данных
np.random.seed(42)
X = 2 * np.random.rand(1000, 3)  # 1000 образцов, 3 признака
y = 4 + 3 * X[:, 0] + 2 * X[:, 1] + 5 * X[:, 2] + np.random.randn(1000)  # Целевая переменная с шумом

# Добавим единицы к признакам для учета смещения (bias)
X_b = np.c_[np.ones((X.shape[0], 1)), X]  # X_b с единичным столбцом

# Параметры
learning_rate = 0.01
n_iterations = 1000
m = len(y)

# Инициализация весов
theta = np.random.randn(X_b.shape[1])

# Обычный градиентный спуск
start_time = time.time()
for iteration in range(n_iterations):
    gradients = np.zeros_like(theta)
    for i in range(m):
        xi = X_b[i, :]
        yi = y[i]
        gradients += 2 * (xi.dot(theta) - yi) * xi
    gradients /= m
    theta -= learning_rate * gradients
time_normal = time.time() - start_time
print(f"Время выполнения обычного градиентного спуска: {time_normal:.4f} секунд")

# Инициализация весов заново
theta = np.random.randn(X_b.shape[1])

# Полный градиентный спуск с матричными операциями
start_time = time.time()
for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta -= learning_rate * gradients
time_matrix = time.time() - start_time
print(f"Время выполнения градиентного спуска с матричными операциями: {time_matrix:.4f} секунд")

print(f"Ускорение за счет использования матричных операций: {time_normal / time_matrix:.2f} раз")
