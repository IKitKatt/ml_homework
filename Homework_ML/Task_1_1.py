from sqlalchemy import create_engine
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Подключение к базе данных
engine = create_engine('postgresql://admin:pgpwd@78.107.239.106:5432/homecrowdb')

# Загружаем данные из таблицы
query = """
SELECT rooms, total_area, living_area, floor, kitchen_area, airports_nearest, last_price, centers_nearest
FROM home_price_task1;
"""
df = pd.read_sql(query, engine)

X = df[['rooms', 'total_area', 'living_area', 'floor', 'kitchen_area', 'airports_nearest', 'centers_nearest']].values
#X = df.drop('last_price', axis=1)
y = df['last_price']

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Нормализуем данные
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Параметры
learning_rate = 0.01  # шаг градиентного спуска
n_iterations = 1000  # количество итераций

# Инициализируем веса и смещение
n_features = X_train.shape[1]
weights = np.zeros(n_features)
bias = 0


# Функция для расчета среднеквадратичной ошибки (MSE)
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# Градиентный спуск
for i in range(n_iterations):
    # Вычисляем предсказания
    y_pred = np.dot(X_train, weights) + bias

    # Вычисляем градиенты
    dw = -(2 / len(y_train)) * np.dot(X_train.T, (y_train - y_pred))
    db = -(2 / len(y_train)) * np.sum(y_train - y_pred)

    # Обновляем веса и смещение
    weights -= learning_rate * dw
    bias -= learning_rate * db

    # Печатаем ошибку каждые 100 итераций
    if i % 100 == 0:
        mse = mean_squared_error(y_train, y_pred)
        print(f"Iteration {i}, MSE: {mse}")

y_pred_test = np.dot(X_test, weights) + bias

# Средняя квадратичная ошибка (MSE)
mse_test = mean_squared_error(y_test, y_pred_test)
print("Test MSE:", mse_test)


plt.scatter(y_test, y_pred_test)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()