import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

df = pd.read_csv("all_v2.csv")

df["date"] = pd.to_datetime(df["date"])

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

#df = df[df['region'] == 2661]
df = df.drop(["date", "region", "time", "level"], axis=1)
df = df[
    (df["price"].quantile(0.1) <= df["price"])
    & (df["price"] <= df["price"].quantile(0.9))
]
df = df[
    (df["area"].quantile(0.05) <= df["area"])
    & (df["area"] <= df["area"].quantile(0.95))
]
df = df[
    (df["kitchen_area"].quantile(0.05) <= df["kitchen_area"])
    & (df["kitchen_area"] <= df["kitchen_area"].quantile(0.95))
]
df = df[df["kitchen_area"] < df["area"]]


print("Датасет обработан")
X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
alpha = 1.0  # Параметр регуляризации (нужно подобрать)
lasso = Lasso(alpha=alpha)
lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Среднеквадратичная ошибка (MSE): {mse}")
print(f"Коэффициент детерминации (R^2): {r2}")


from sklearn.model_selection import GridSearchCV
param_grid = {'alpha': np.logspace(-3, 3, 7)}  #Диапазон значений alpha
grid_search = GridSearchCV(Lasso(), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_alpha = grid_search.best_params_['alpha']
best_lasso = grid_search.best_estimator_
print(f"Лучшее значение alpha: {best_alpha}")

y_pred_best = best_lasso.predict(X_test)
mse_best = mean_squared_error(y_test, y_pred_best)
mae_best = mean_absolute_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)

print(f"MSE с лучшим alpha: {mse_best}")
print(f"MAE с лучшим alpha: {mae_best}")
print(f"R^2 с лучшим alpha: {r2_best}")