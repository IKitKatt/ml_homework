from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import time

# Подключение к базе данных PostgreSQL
engine = create_engine('postgresql://admin:pgpwd@78.107.239.106:5432/homecrowdb')

# Загрузка данных
query = 'SELECT * FROM autos'
data = pd.read_sql(query, engine)

# Предобработка данных
data = data.dropna(subset=['price'])  # Удаление записей с пропущенными значениями в цене
data = data[(data['price'] > 500) & (data['price'] < 100000)]  # Удаление аномальных значений цены

# Ограничиваем количество уникальных значений для категориальных признаков
categorical_columns = data.select_dtypes(include=['object']).columns
top_n = 10  # Оставляем только 10 популярных значений

for col in categorical_columns:
    top_categories = data[col].value_counts().nlargest(top_n).index
    data[col] = data[col].apply(lambda x: x if x in top_categories else 'other')

# Применяем one-hot encoding с уменьшенным количеством категорий
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Проверка, что целевая переменная существует и нет категориальных данных
if 'price' not in data.columns:
    raise KeyError("Целевая переменная 'price' отсутствует в данных")

X = data.drop('price', axis=1)
y = data['price']

# Масштабирование признаков и целевой переменной
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()  # Преобразование в 1D массив

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Реализация стохастического градиентного спуска
def stochastic_gradient_descent(X, y, batch_size, learning_rate, n_iterations, tolerance):
    m, n = X.shape
    theta = np.random.randn(n) * 0.01  # Инициализация весов с малым значением
    mse_history = []
    start_time = time.time()

    for iteration in range(n_iterations):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, m, batch_size):
            xi = X_shuffled[i:i + batch_size]
            yi = y_shuffled[i:i + batch_size]

            # Проверка на пустые массивы
            if len(xi) == 0 or len(yi) == 0:
                continue

            gradients = 2 / len(xi) * xi.T.dot(xi.dot(theta) - yi)
            theta -= learning_rate * gradients

            # Проверка на переполнение
            if np.isnan(theta).any() or np.isinf(theta).any():
                print("Переполнение в theta, уменьшите learning_rate.")
                return theta, iteration, time.time() - start_time, mse_history

        # Вычисление MSE и проверка сходимости
        mse = mean_squared_error(y, X.dot(theta))
        mse_history.append(mse)

        if len(mse_history) > 1 and abs(mse_history[-2] - mse_history[-1]) < tolerance:
            break

    end_time = time.time()
    return theta, iteration + 1, end_time - start_time, mse_history


# Исследование сходимости SGD в зависимости от размера батча
batch_sizes = np.arange(5, 500, 10)
learning_rate = 0.001  # Уменьшили скорость обучения для устойчивости
n_iterations = 1000
tolerance = 1e-3
results = []

for batch_size in batch_sizes:
    theta, steps, elapsed_time, mse_history = stochastic_gradient_descent(X_train, y_train, batch_size, learning_rate,
                                                                          n_iterations, tolerance)
    results.append((batch_size, steps, elapsed_time, mse_history))

# Построение графиков
batch_sizes_plot = [result[0] for result in results]
steps_to_convergence = [result[1] for result in results]
times_to_convergence = [result[2] for result in results]

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(batch_sizes_plot, steps_to_convergence, marker='o')
plt.xlabel('Batch Size')
plt.ylabel('Steps to Convergence')
plt.title('Steps to Convergence vs Batch Size')

plt.subplot(1, 2, 2)
plt.plot(batch_sizes_plot, times_to_convergence, marker='o', color='orange')
plt.xlabel('Batch Size')
plt.ylabel('Time to Convergence (seconds)')
plt.title('Time to Convergence vs Batch Size')
plt.show()
