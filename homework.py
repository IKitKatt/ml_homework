from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import json
import matplotlib.pyplot as plt


with open('auth.json', 'r') as f:
    config = json.load(f)

db_url = f"postgresql+psycopg2://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
engine = create_engine(db_url)


#price, yr_built, bathrooms, bedrooms, floors, grade, condition
query = ("SELECT price, yr_built, bathrooms, bedrooms, floors, grade, condition FROM house_data WHERE yr_built >= 2000")
df = pd.read_sql_query(query, engine)

X = df[['yr_built']]
Y = df['price']

# lr = LinearRegression()
# lr.fit(df[['yr_built']], df['price'])
#
# df.plot(kind="hist",x="price",y="yr_built", bins=5)
# plt.show()

lin_reg = LinearRegression()
lin_reg.fit(X, Y)

# Прогнозы и коэффициент детерминации для линейной модели
Y_pred_linear = lin_reg.predict(X)
r2_linear = r2_score(Y, Y_pred_linear)
r = np.sqrt(r2_linear)
#
# print(f"Коэффициент детерминации (R^2) для линейной регрессии: {r2_linear:.4f}")
# print(f"Коэффициент корреляции (R) для линейной регрессии: {r:.4f}")
#
#
# plt.scatter(X, Y, color='gray', label='Данные')  # Исходные данные
# plt.plot(X, Y_pred_linear, color='red', label=f'Линейная регрессия (R^2 = {r2_linear:.4f})')  # Линия регрессии
# plt.legend()
# plt.show()

# Полиномиальная регрессия для разных степеней
best_degree = 1
best_r2 = r2_linear
degrees = [2, 3, 4, 5]

for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    poly_reg = LinearRegression()
    poly_reg.fit(X_poly, Y)

    Y_pred_poly = poly_reg.predict(X_poly)
    r2_poly = r2_score(Y, Y_pred_poly)

    print(f"Коэффициент детерминации (R^2) для полинома степени {degree}: {r2_poly:.4f}")

    if r2_poly > best_r2:
        best_r2 = r2_poly
        best_degree = degree

print(f"\nНаилучшая степень полинома: {best_degree}, с коэффициентом детерминации: {best_r2:.4f}")
print(f"Коэффициент корреляции (R) для линейной регрессии: {r:.4f}")

# Визуализация данных и наилучшей регрессии
plt.scatter(X, Y, color='blue', label='Данные')

# Линейная регрессия
plt.plot(X, Y_pred_linear, color='red', label=f'Линейная регрессия (R^2 = {r2_linear:.4f})')

# Полиномиальная регрессия для наилучшей степени
poly = PolynomialFeatures(degree=best_degree)
X_poly = poly.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, Y)
Y_pred_best_poly = poly_reg.predict(X_poly)
plt.plot(X, Y_pred_best_poly, color='green', label=f'Полином степени {best_degree} (R^2 = {best_r2:.4f})')

plt.legend()
plt.title('Регрессия: Линейная и Полиномиальная')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()