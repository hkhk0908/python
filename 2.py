import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. 손글씨 숫자 데이터 로드
digits = load_digits()
X = digits.data
y = digits.target

# 2. 데이터를 훈련 데이터와 테스트 데이터로 분할 (80% 훈련, 20% 테스트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 소프트맥스 회귀 모델 (multinomial logistic regression) 정의 및 학습
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)

# 4. 예측
y_pred = model.predict(X_test)

# 5. 평가: 정확도, 분류 보고서, 혼동 행렬 출력
print("정확도:", accuracy_score(y_test, y_pred))
print("\n분류 보고서:\n", classification_report(y_test, y_pred))
print("혼동 행렬:\n", confusion_matrix(y_test, y_pred))