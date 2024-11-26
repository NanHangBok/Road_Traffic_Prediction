"""
XGBoost with Sliding Window
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import matplotlib.dates as mdates


plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

# 데이터 불러오기
recent_data = pd.read_csv('4.csv', parse_dates=['ds'])

# 추가 특성 생성 (요일과 시간)
recent_data['day_of_week'] = recent_data['ds'].dt.dayofweek
recent_data['hour'] = recent_data['ds'].dt.hour

# Windowing 설정
window_size = 24  # 과거 24시간을 입력으로 사용

# 윈도우 생성 함수
def create_windowed_data(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        # 입력 데이터 (과거 24시간)
        features = data[['y', 'Temperature', 'Rainfall', 'WindSpeed', 'day_of_week', 'hour']].iloc[i:i+window_size].values.flatten()
        X.append(features)
        # 출력 데이터 (다음 1시간 교통량)
        y.append(data['y'].iloc[i+window_size])
    return np.array(X), np.array(y)

# 윈도우 데이터 생성
X, y = create_windowed_data(recent_data, window_size)

# 데이터셋 분할
train_end = int(len(X) * 0.7)
validation_end = int(len(X) * 0.85)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:validation_end], y[train_end:validation_end]
X_test, y_test = X[validation_end:], y[validation_end:]

# XGBoost 모델 설정 및 학습
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=5,
    early_stopping_rounds=20,
    eval_metric='rmse'
)
xgb_model.fit(
    X_train, y_train, 
    eval_set=[(X_val, y_val)], 
    verbose=True,

)
# 학습 후 evals_result_에서 RMSE 값 추출
evals_result = xgb_model.evals_result()

# 마지막 RMSE 값 가져오기
last_rmse = evals_result['validation_0']['rmse'][-1]
print(f"최종 RMSE: {last_rmse}")


# 예측
train_pred = xgb_model.predict(X_train)
val_pred = xgb_model.predict(X_val)
test_pred = xgb_model.predict(X_test)

# 평가
print("XGBoost Validation MAE:", mean_absolute_error(y_val, val_pred))
print("XGBoost Validation RMSE:", mean_squared_error(y_val, val_pred, squared=False))
print("XGBoost Test MAE:", mean_absolute_error(y_test, test_pred))
print("XGBoost Test RMSE:", mean_squared_error(y_test, test_pred))

# 시각화 (테스트 데이터 예측 vs 실제값 - 일주일 범위)
plt.figure(figsize=(14, 6))
plt.plot(recent_data['ds'].iloc[validation_end + window_size:], y_test, label="실제 Test Data", color="black")
plt.plot(recent_data['ds'].iloc[validation_end + window_size:], test_pred, label="예측한 Test Data", color="blue")
plt.title("도로교통량 예측")
plt.xlabel("날짜")
plt.ylabel("교통량")
plt.legend()

# x축을 가장 최신 날짜부터 일주일로 설정
end_date = recent_data['ds'].max()  # 최신 날짜
start_date = end_date - pd.Timedelta(days=7)  # 최신 날짜 - 7일
plt.xlim([start_date, end_date])

# 날짜 포맷 설정 (날짜와 시간 표시)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))  # 1일 간격으로 표시

plt.show()

# 시각화 (테스트 데이터 예측 vs 실제값 - 하루 범위)
plt.figure(figsize=(14, 6))
plt.plot(recent_data['ds'].iloc[validation_end + window_size:], y_test, label="실제 Test Data", color="black")
plt.plot(recent_data['ds'].iloc[validation_end + window_size:], test_pred, label="예측한 Test Data", color="blue")
plt.title("도로교통량 예측 범위-하루")
plt.xlabel("날짜")
plt.ylabel("교통량")
plt.legend()

# x축을 가장 최신 날짜부터 하루로 설정
start_date = end_date - pd.Timedelta(days=1)  # 최신 날짜 - 1일
plt.xlim([start_date, end_date])

# 날짜 포맷 설정 (날짜와 시간 표시)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))  # 1시간 간격으로 표시

plt.show()

# 미래 예측을 위해 테스트 데이터의 마지막 윈도우를 가져오기
last_window = X_test[-1].reshape(1, -1)  # 가장 최근 윈도우를 2D 형식으로 변환

# 미래 예측을 저장할 리스트
future_steps = 24  # 예측할 미래 시간 (예: 24시간)
future_preds = []

for _ in range(future_steps):
    # 현재 윈도우에 대한 예측
    future_pred = xgb_model.predict(last_window)[0]
    future_preds.append(future_pred)
    
    # 새로운 데이터를 위한 윈도우 업데이트
    # last_window에서 옛 데이터를 제거하고, 예측값과 외부 변수를 추가하여 윈도우를 이동
    last_window = np.roll(last_window, shift=-len(last_window[0])//window_size, axis=1)  # 좌측으로 윈도우 이동
    last_window[0, -len(last_window[0])//window_size] = future_pred  # 예측된 값을 마지막 위치에 추가
    
    # 외부 변수를 고정 값으로 사용하여 윈도우를 채우기
    # 예시로 마지막 테스트 데이터의 외부 변수를 사용
    last_window[0, -len(last_window[0])//window_size + 1:] = X_test[-1, -len(last_window[0])//window_size + 1:]

# 미래 예측값을 시각화
future_dates = [recent_data['ds'].iloc[-1] + pd.Timedelta(hours=i + 1) for i in range(future_steps)]

# 시각화 (테스트 데이터 예측 및 미래 예측 - 마지막 구간만 보기)
plt.figure(figsize=(14, 6))
plt.plot(recent_data['ds'].iloc[validation_end + window_size:], y_test, label="실제 Test Data", color="black")
plt.plot(recent_data['ds'].iloc[validation_end + window_size:], test_pred, label="예측한 Test Data", color="blue")
plt.plot(future_dates, future_preds, label="예측된 Future Data", color="orange", linestyle="dashed")
plt.title("도로교통량 예측 및 미래 예측 (마지막 구간)")

# x축을 가장 최신 날짜부터 일주일로 설정
end_date = future_dates[-1]  # 예측된 미래 데이터의 마지막 날짜
start_date = end_date - pd.Timedelta(days=7)  # 마지막 날짜 - 7일
plt.xlim([start_date, end_date])

plt.xlabel("날짜")
plt.ylabel("교통량")
plt.legend()

# 날짜 포맷 설정 (날짜와 시간 표시)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))  # 1일 간격으로 표시

plt.show()

# 시각화 (테스트 데이터 예측 및 미래 예측 - 마지막 1시간과 예측 부분만 보기)
plt.figure(figsize=(14, 6))
plt.plot(recent_data['ds'].iloc[validation_end + window_size:], y_test, label="실제 Test Data", color="black")
plt.plot(recent_data['ds'].iloc[validation_end + window_size:], test_pred, label="예측한 Test Data", color="blue")
plt.plot(future_dates, future_preds, label="예측된 Future Data", color="orange", linestyle="dashed")
plt.title("도로교통량 예측 및 미래 예측 (실제 데이터 마지막 1시간과 예측)")

# x축을 실제 데이터의 마지막 1시간과 그 이후 예측 구간으로 설정
end_date = future_dates[-1]  # 예측된 미래 데이터의 마지막 날짜
start_date = recent_data['ds'].iloc[validation_end + window_size:].iloc[-1] - pd.Timedelta(hours=1)  # 실제 데이터의 마지막 1시간
plt.xlim([start_date, end_date])

plt.xlabel("날짜")
plt.ylabel("교통량")
plt.legend()

# 날짜 포맷 설정 (시간 표시)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))  # 1시간 간격으로 표시

plt.show()