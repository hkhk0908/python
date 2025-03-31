import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score

# 데이터 로딩 및 전처리
data = fetch_california_housing()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 모델 훈련 및 평가 함수
def evaluate_model(model, X_train, X_test, y_train, y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    elapsed_time = time.time() - start_time

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2, elapsed_time


# 모델별 성능 저장
evaluation_results = {}

# 실습 1: 선형 회귀 모델
lin_reg = LinearRegression()
evaluation_results['Linear Regression'] = evaluate_model(lin_reg, X_train, X_test, y_train, y_test)

# 실습 2: Lasso 회귀 모델
lasso = Lasso(alpha=0.1)
evaluation_results['Lasso Regression'] = evaluate_model(lasso, X_train, X_test, y_train, y_test)

# 실습 3: Ridge 회귀 모델
ridge = Ridge(alpha=0.1)
evaluation_results['Ridge Regression'] = evaluate_model(ridge, X_train, X_test, y_train, y_test)

# 실습 4: 다항 회귀 모델
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_reg = LinearRegression()
evaluation_results['Polynomial Regression (degree=2)'] = evaluate_model(poly_reg, X_train_poly, X_test_poly, y_train,
                                                                        y_test)

# 결과 출력
for model_name, (mse, r2, elapsed_time) in evaluation_results.items():
    print(f"{model_name}: MSE={mse:.4f}, R²={r2:.4f}, Time={elapsed_time:.4f}s")

# 모델별 성능 비교 시각화
models = list(evaluation_results.keys())
mse_values = [evaluation_results[m][0] for m in models]
r2_values = [evaluation_results[m][1] for m in models]
times = [evaluation_results[m][2] for m in models]

fig, ax = plt.subplots(1, 3, figsize=(18, 5))
ax[0].bar(models, mse_values, color='b')
ax[0].set_title('Mean Squared Error')
ax[0].set_xticklabels(models, rotation=45, ha='right')

ax[1].bar(models, r2_values, color='g')
ax[1].set_title('R² Score')
ax[1].set_xticklabels(models, rotation=45, ha='right')

ax[2].bar(models, times, color='r')
ax[2].set_title('Training Time (seconds)')
ax[2].set_xticklabels(models, rotation=45, ha='right')

plt.tight_layout()
plt.show()
