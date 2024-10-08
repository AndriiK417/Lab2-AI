import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

# 1. Генерація випадкового набору даних
np.random.seed(58)
X = np.random.rand(1000, 1) * 100
# y = np.random.rand(1000, 1) * 100
y = 0 * X.squeeze() + np.random.randn(1000) * 50

# Відображення згенерованих даних до нормалізації
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', alpha=0.6)
plt.title('Згенеровані дані до нормалізації')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.xlim(0, 100)
plt.show()

# 2. Нормалізація значень
scaler = MinMaxScaler()
X_scaled = X
# Y_scaled = y = 5 * X.squeeze() + np.random.randn(1000) * 10
Y_scaled = scaler.fit_transform(y.reshape(-1, 1))
# візуалізація
plt.figure(figsize=(10, 6))
plt.scatter(X_scaled, Y_scaled, color='blue', alpha=0.6)
plt.title('Згенеровані дані після нормалізації')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.xlim(0, 100)
plt.ylim(-4, 5)
plt.show()

# 3. Розділення на навчальну і тестову вибірки (80% на навчальну, 20% на тестову)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)

# 4. Візуалізація навчальної і тестової вибірок
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='#008000', label='Training Data (Навчальна вибірка)', alpha=0.7)  # зелений
plt.scatter(X_test, y_test, color='#FF0000', label='Test Data (Тестова вибірка)', alpha=0.5)  # червоний
plt.title('Розділення вибірки на навчальну і тестову')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()

# 4,5. Візуалізація навчальної і тестової вибірок після масштабування
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='#008000', label='Training Data (Навчальна вибірка)', alpha=0.7)  # зелений
plt.scatter(X_test, y_test, color='#FF0000', label='Test Data (Тестова вибірка)', alpha=0.5)  # червоний
plt.title('Навчальна та тестова вибірки')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.xlim(0, 100)
plt.ylim(-4, 5)
plt.show()

# 5. Навчання KNN-регресора з різними значеннями К
k_values = range(1, 800)
errors = []

for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    error = mean_squared_error(y_test, y_pred)
    errors.append(error)

# 6. Вибір найкращого К з найменшою похибкою
best_k = k_values[np.argmin(errors)]
print(f"Найкраще значення K: {best_k}")

# 7. Візуалізація похибки для різних K
plt.figure(figsize=(10, 6))
plt.plot(k_values, errors, marker='o', linestyle='-', color='b', label='Похибка')
plt.axvline(x=best_k, color='r', linestyle='--', label=f'Значення К з найменшою похибкою: {best_k}')
plt.title('Графік похибки для різних значень К')
plt.xlabel('Значення К')
plt.ylabel('Похибка')
plt.legend()
min_error = min(errors)
plt.text(best_k + 10, min_error + 0.005, f'{min_error:.4f}', color='red', fontsize=12)
plt.grid(True)
plt.show()

# 8. Візуалізація результатів з найкращим K
knn_best = KNeighborsRegressor(n_neighbors=best_k)
knn_best.fit(X_train, y_train)
y_pred_best = knn_best.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Дані тестової вибірки')
plt.scatter(X_test, y_pred_best, color='red', label='Прогнозовані дані, отримані за допомогою KNN-регресора')
plt.title(f'Регресія KNN для значення К = {best_k}')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()

# 8. Візуалізація результатів з найкращим K (після масштабування)
knn_best = KNeighborsRegressor(n_neighbors=best_k)
knn_best.fit(X_train, y_train)
y_pred_best = knn_best.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Дані тестової вибірки')
plt.scatter(X_test, y_pred_best, color='red', label='Прогнозовані дані, отримані за допомогою KNN-регресора')
plt.title(f'Регресія KNN для значення К = {best_k}')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.xlim(0, 100)
plt.ylim(-4, 5)
plt.show()

# 9. Обчислення похибок (MSE і MAE) для найкращого K
mse = mean_squared_error(y_test, y_pred_best)
mae = mean_absolute_error(y_test, y_pred_best)

print(f"Середньоквадратична похибка (MSE): {mse:.4f}")

