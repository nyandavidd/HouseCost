import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump, load
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split


def optimize_correlation(df, target_column="price"):
    """Находит линейную комбинацию признаков, которая максимально коррелирует с заданным целевым признаком."""
    features = df.drop(columns=target_column).columns
    X = df[features].values
    P = df[target_column].values
    n = len(features)

    cov_matrix = np.cov(X, rowvar=False)
    cov_XP = np.cov(X, P, rowvar=False)[:n, n:].flatten()

    def target_function(a_rest):
        weights = np.insert(a_rest, 0, 1.0)
        L = weights @ cov_XP
        S = weights @ cov_matrix @ weights
        return -np.abs(L) / np.sqrt(S)  # Минимизируем -|L|/sqrt(S)

    initial_guess = np.zeros(n - 1)
    result = minimize(target_function, initial_guess)
    weights = np.insert(result.x, 0, 1.0)
    linear_combination = X @ weights
    correlation = np.corrcoef(linear_combination, P)[0, 1]
    return correlation, weights
df = pd.read_csv("all_v2.csv")

df["date"] = pd.to_datetime(df["date"])

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

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

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

target_train = df_train["price"]
features_train = df_train.drop(["price"], axis=1)

target_test = df_test["price"]
features_test = df_test.drop(["price"], axis=1)


X_train = df_train.drop(
    [
        "price",
    ],
    axis=1,
)
y_train = target_train

X_test = df_test.drop(
    [
        "price",
    ],
    axis=1,
)
y_test = target_test

n_estimators = 10
model = RandomForestRegressor(
    n_estimators=n_estimators,
    random_state=42,
    n_jobs=14,
)
model.fit(X_train, y_train)
print(X_train.columns)
print("Научил лес")
predictions = model.predict(X_test)
print("Предсказание с учётом инфляции:")
rmse = root_mean_squared_error(target_test, predictions)
mae = mean_absolute_error(target_test, predictions)
r2 = r2_score(target_test, predictions)
rmse_relative = rmse / target_test.mean()
mae_relative = mae / target_test.mean()
mape = np.mean(np.abs((predictions - target_test) / target_test))
smape = np.mean(np.abs(predictions - target_test)) / np.mean(predictions + target_test)
results = {
    "rmse": rmse,
    "rmse_relative": rmse_relative,
    "mae": mae,
    "mae_relative": mae_relative,
    "mape": mape,
    "smape": smape,
    "r2": r2,
}

print(n_estimators,"деревьев,",)
for el in results:
    print(el, ':', results[el])
