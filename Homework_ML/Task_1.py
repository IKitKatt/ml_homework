import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Шаг 1: Подключение к базе данных и загрузка данных
conn = psycopg2.connect(
    dbname="homecrowdb", user="admin", password="pgpwd", host="78.107.239.106", port="5432"
)
query = "SELECT rooms, total_area, living_area, floor, kitchen_area, airports_nearest, last_price FROM home_price_task1"
data = pd.read_sql(query, conn)
conn.close()

# Шаг 2: Нормализация данных (для ускорения градиентного спуска)
data = (data - data.mean()) / data.std()

# Выделяем входные переменные (X) и целевую переменную (y)
X = data[['rooms', 'total_area', 'living_area', 'floor', 'kitchen_area', 'airports_nearest']].values
y = data['last_price'].values

# Добавляем единичный столбец для учета смещения
X = np.c_[np.ones(X.shape[0]), X]


# Функция для расчета MSE
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# Градиентный спуск
def gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    m, n = X.shape
    theta = np.zeros(n)  # Инициализация весов
    errors = []

    for epoch in range(epochs):
        # Прогноз модели
        y_pred = X.dot(theta)

        # Расчет ошибки и градиента
        error = y_pred - y
        gradient = X.T.dot(error) / m
        theta -= learning_rate * gradient  # Обновление весов

        # Запись ошибки для визуализации
        errors.append(mse(y, y_pred))

    return theta, errors


# Шаг 3: Запуск градиентного спуска
learning_rate = 0.01
epochs = 1000
theta, errors = gradient_descent(X, y, learning_rate, epochs)

# Шаг 4: Визуализация ошибки
plt.plot(errors)
plt.xlabel('Итерация')
plt.ylabel('Среднеквадратичная ошибка (MSE)')
plt.title('График обучения')
plt.show()

# Оценка финальной ошибки
final_mse = mse(y, X.dot(theta))
print(f"Среднеквадратичная ошибка модели: {final_mse}")