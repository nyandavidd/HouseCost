import pickle as pkl
import time

import matplotlib.pyplot as plt

# # 2. Create the Heatmap
# plt.figure(figsize=(10, 8))  # Adjust size as needed
# sns.heatmap(
#     correlation_matrix,
#     annot=True,  # Show correlation values
#     cmap="coolwarm",  # Use a diverging color palette
#     fmt=".2f",  # Format correlation values
#     linewidths=0.5,
# )
# plt.title("Correlation Matrix of Real Estate Features")
# plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for readability
# plt.tight_layout()
# plt.show()
# кросс-валидация
import pandas as pd
import seaborn as sns

from joblib import dump, load
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

# 3. Feature Importance from a Model (Example with Linear Regression)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score  # Или другая подходящая метрика
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
    silhouette_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

print("Загрузка данных...")
df = pd.read_csv("all_v2.csv")
print(df)


print("Данные загружены!")
df = df[df["region"] == 2661]
df = df[df["price"] > 1_000_000]
df = df[df["price"] < 50_000_000]

print("Подготовка данных...")

df = df.drop(
    [
        "day",
        "minute",
        "hour",
    ],
    axis=1,
)
features = df[df.columns]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
features_scaled = pd.DataFrame(features_scaled, columns=features.columns)

def forest_check(
    n_clusters,
    data=df,
    features_scaled=features_scaled,
):  
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    start_time = time.time()  
    kmeans.fit(features_scaled)
    end_time = time.time() 
    print(
        f"  Данные кластеризованы! Время обучения: {end_time - start_time:.2f} секунд"
    )
    df = data
    df["cluster"] = kmeans.labels_
    df = df.join(
        df.groupby("cluster").agg(
            mean_price=("price", "mean"),
            median_price=("price", "median"),
            mean_price_per_sqm=("price", lambda x: (x / df["area"]).mean()),
            median_price_per_sqm=(
                "price",
                lambda x: (x / df["area"]).median(),
            ),
        ),
        on="cluster",
    )
    df["mean_price_per_sqm"] *= df["area"]
    df["median_price_per_sqm"] *= df["area"]
  
    X = df.drop(["price"], axis=1)
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=20,
        max_features="log2",
        random_state=42,
        n_jobs=-1,
    )
    # 4. Обучение и оценка каждой модели
    # print(f"Модель: Random Forest")
    print("  Обучение модели...")
    start_time = time.time()  
    model.fit(X_train, y_train)
    end_time = time.time()  
    print(f"  Модель обучена! Время обучения: {end_time - start_time:.2f} секунд")

    y_pred = model.predict(X_test)

    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    medae = median_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    price_stats = df["price"].describe()
    print("Максимум:", price_stats["max"])
    print("Минимум:", price_stats["min"])
    print("Среднее:", price_stats["mean"])
    print("Медиана:", price_stats["50%"])  # 50% квантиль - это медиана
    print("Стандартное отклонение:", df["price"].std())
    print("-" * 20)
    print(f"Число кластеров: {n_clusters}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    # print(f"  MedAE: {medae:.2f}")
    print(f"  R^2: {r2:.2f}")
    
    print("-" * 20)
    dump(model, 'model24.joblib') 
    df.to_csv('spb_map.csv')

forest_check(24)
