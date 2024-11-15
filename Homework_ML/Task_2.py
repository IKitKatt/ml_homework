from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Подключение к базе данных
engine = create_engine('postgresql://admin:pgpwd@78.107.239.106:5432/homecrowdb')

# Загружаем данные
query = """
SELECT x1, x2, x3
FROM x_samples;
"""
df = pd.read_sql(query, engine)

# Отделяем признаки и целевую переменную
X = df[['x2', 'x3']]
y = df['x1']

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Нормализация данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Линейная регрессия с градиентным спуском
learning_rate = 0.01
n_iterations = 1000
weights = np.zeros(X_train.shape[1])
bias = 0

# Градиентный спуск
for i in range(n_iterations):
    y_pred = np.dot(X_train, weights) + bias
    dw = -(2 / len(y_train)) * np.dot(X_train.T, (y_train - y_pred))
    db = -(2 / len(y_train)) * np.sum(y_train - y_pred)
    weights -= learning_rate * dw
    bias -= learning_rate * db

# Предсказания и визуализация
y_pred_test_linear = np.dot(X_test, weights) + bias
mse_test_linear = np.mean((y_test - y_pred_test_linear) ** 2)

import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred_test_linear, color="blue")
plt.xlabel("Actual Target Values")
plt.ylabel("Predicted Target Values (Linear)")
plt.title("Linear Regression Predictions vs Actual")
plt.show()

degrees = [1, 2, 3, 4, 5]  # Разные степени для подбора оптимальной
mse_values = []

for degree in degrees:
    poly = PolynomialFeatures(degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_pred_test_poly = model.predict(X_test_poly)
    mse = mean_squared_error(y_test, y_pred_test_poly)
    mse_values.append(mse)

    plt.scatter(y_test, y_pred_test_poly, label=f'Degree {degree}')
    plt.xlabel("Actual Target Values")
    plt.ylabel("Predicted Target Values")
    plt.legend()
    plt.title(f"Polynomial Regression Predictions vs Actual (Degree {degree})")
    plt.show()

# Оптимальная степень
optimal_degree = degrees[np.argmin(mse_values)]
print(f"Optimal polynomial degree: {optimal_degree}")
print(f"MSE values for degrees: {dict(zip(degrees, mse_values))}")