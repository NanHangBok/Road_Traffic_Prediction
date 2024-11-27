"""
XGBoost with Sliding Window
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import matplotlib.dates as mdates

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 불러오기
recent_data = pd.read_csv('4.csv', parse_dates=['ds'])

# 추가 특성 생성 (요일과 시간)
recent_data['day_of_week'] = recent_data['ds'].dt.dayofweek
recent_data['hour'] = recent_data['ds'].dt.hour

# 윈도우 생성 함수
def create_windowed_data(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        features = data[['y', 'Temperature', 'Rainfall', 'WindSpeed', 'day_of_week', 'hour']].iloc[i:i + window_size].values.flatten()
        X.append(features)
        y.append(data['y'].iloc[i + window_size])
    return np.array(X), np.array(y)

# Windowing 설정
window_size = 24  # 과거 24시간을 입력으로 사용
X, y = create_windowed_data(recent_data, window_size) # 윈도우 데이터 생성

# 데이터셋 분할 
train_end = int(len(X) * 0.7)
validation_end = int(len(X) * 0.85)

# (70% Train, 15% Validation, 15% Test)
X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:validation_end], y[train_end:validation_end]
X_test, y_test = X[validation_end:], y[validation_end:]

# XGBoost 모델 설정 및 학습
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',  # 손실 함수로 평균 제곱 오차 사용
    n_estimators=1000,  # 트리의 개수
    learning_rate=0.01,  # 학습률
    max_depth=5,  # 트리 깊이
    early_stopping_rounds=20,  # 얼리스탑 20회 (조기종료)
    eval_metric='rmse'  # 평가 RMSE 사용
)

xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

# 학습 후 evals_result_에서 RMSE 값 추출
evals_result = xgb_model.evals_result()
last_rmse = evals_result['validation_0']['rmse'][-1]
print(f"Final RMSE: {last_rmse}")

# 예측
train_pred = xgb_model.predict(X_train)
val_pred = xgb_model.predict(X_val)
test_pred = xgb_model.predict(X_test)

# 검증 및 테스트 평가
print("XGBoost Validation MAE:", mean_absolute_error(y_val, val_pred))
print("XGBoost Validation RMSE:", mean_squared_error(y_val, val_pred, squared=False))
print("XGBoost Test MAE:", mean_absolute_error(y_test, test_pred))
print("XGBoost Test RMSE:", mean_squared_error(y_test, test_pred))

# 시각화 (범위 일주일)
plt.figure(figsize=(14, 6))
end_date = recent_data['ds'].max()
start_date = end_date - pd.Timedelta(days=7)
plt.plot(recent_data['ds'].iloc[validation_end + window_size:], y_test, label="실제 테스트 데이터", color="black")
plt.plot(recent_data['ds'].iloc[validation_end + window_size:], test_pred, label="예측된 테스트 데이터", color="blue")
plt.title("도로 교통량 예측 - 1주일 범위")
plt.xlabel("날짜")
plt.ylabel("교통량")
plt.legend()
plt.xlim([start_date, end_date])
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
plt.show()

# 시각화 (범위 하루)
plt.figure(figsize=(14, 6))
start_date = end_date - pd.Timedelta(days=1)
plt.plot(recent_data['ds'].iloc[validation_end + window_size:], y_test, label="실제 테스트 데이터", color="black")
plt.plot(recent_data['ds'].iloc[validation_end + window_size:], test_pred, label="예측된 테스트 데이터", color="blue")
plt.title("도로 교통량 예측 - 1일 범위")
plt.xlabel("날짜")
plt.ylabel("교통량")
plt.legend()
plt.xlim([start_date, end_date])
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
plt.show()

# 미래 예측
last_window = X_test[-1].reshape(1, -1)
future_steps = 24  # 예측할 미래 시간 (24시간)
future_preds = []

for _ in range(future_steps):
    future_pred = xgb_model.predict(last_window)[0]
    future_preds.append(future_pred)
    last_window = np.roll(last_window, shift=-len(last_window[0]) // window_size, axis=1)
    last_window[0, -len(last_window[0]) // window_size] = future_pred
    last_window[0, -len(last_window[0]) // window_size + 1:] = X_test[-1, -len(last_window[0]) // window_size + 1:]

future_dates = [recent_data['ds'].iloc[-1] + pd.Timedelta(hours=i + 1) for i in range(future_steps)]

# 미래 예측 시각화
plt.figure(figsize=(14, 6))
plt.plot(recent_data['ds'].iloc[validation_end + window_size:], y_test, label="실제 테스트 데이터", color="black")
plt.plot(recent_data['ds'].iloc[validation_end + window_size:], test_pred, label="예측된 테스트 데이터", color="blue")
plt.plot(future_dates, future_preds, label="미래 예측 데이터", color="orange", linestyle="dashed")
plt.title("도로 교통량 예측 및 미래 예측")
plt.xlabel("날짜")
plt.ylabel("교통량")
plt.legend()
start_date = recent_data['ds'].iloc[validation_end + window_size:].iloc[-1] - pd.Timedelta(hours=1)
plt.xlim([start_date, future_dates[-1]])
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
plt.show()