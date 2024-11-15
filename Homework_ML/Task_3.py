import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
X1 = np.random.rand(100, 1) * 10
X2 = np.random.rand(100, 1) * 10
X3 = np.random.rand(100, 1) * 10
y = 3 * (X1**2) + 2 * (X2**3) - 5 * X3 + np.random.randn(100, 1) * 10

# Объединяем в DataFrame
data = pd.DataFrame(np.hstack([X1, X2, X3, y]), columns=['X1', 'X2', 'X3', 'y'])

# Инициализация параметров
learning_rate = 0.01
n_iterations = 1000
X = data[['X1', 'X2', 'X3']].values
y = data['y'].values
m, n = X.shape

# Добавляем столбец единиц для смещения (bias)
X_b = np.c_[np.ones((m, 1)), X]
theta = np.random.randn(n + 1)  # Инициализация весов случайными значениями

# Сохраняем историю значений функции потерь для визуализации
loss_history = []

# Градиентный спуск
for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - learning_rate * gradients
    loss = np.mean((X_b.dot(theta) - y) ** 2)
    loss_history.append(loss)

# Визуализация траектории функции потерь
plt.plot(loss_history)
plt.xlabel("Iteration")
plt.ylabel("Mean Squared Error")
plt.title("Gradient Descent Trajectory")
plt.show()

learning_rates = [0.001, 0.01, 0.1, 0.5]
n_iterations = 1000

# Визуализация для каждого значения learning_rate
plt.figure(figsize=(10, 7))

for lr in learning_rates:
    theta = np.random.randn(n + 1)
    loss_history = []
    for iteration in range(n_iterations):
        gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - lr * gradients
        loss = np.mean((X_b.dot(theta) - y) ** 2)
        loss_history.append(loss)

    plt.plot(loss_history, label=f"Learning rate {lr}")

plt.xlabel("Iteration")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.title("Influence of Learning Rate on Convergence")
plt.show()

# Масштабирование данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_b_scaled = np.c_[np.ones((m, 1)), X_scaled]

# Градиентный спуск на масштабированных данных
learning_rate = 0.01
theta = np.random.randn(n + 1)
loss_history_scaled = []

for iteration in range(n_iterations):
    gradients = 2/m * X_b_scaled.T.dot(X_b_scaled.dot(theta) - y)
    theta = theta - learning_rate * gradients
    loss = np.mean((X_b_scaled.dot(theta) - y) ** 2)
    loss_history_scaled.append(loss)

# Визуализация сходимости на масштабированных данных
plt.plot(loss_history, label="Without Scaling")
plt.plot(loss_history_scaled, label="With Scaling")
plt.xlabel("Iteration")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.title("Effect of Feature Scaling on Convergence")
plt.show()
