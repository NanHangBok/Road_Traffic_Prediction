"""
Prophet
No Variable (변수없음)
"""

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 불러오기
recent_data = pd.read_csv('4.csv', parse_dates=['ds'])

# 데이터 전처리
data_pret = recent_data[['ds', 'y']].copy()

# 데이터 분할 (70% Train, 15% Validation, 15% Test)
train_end = int(len(data_pret) * 0.7)
validation_end = int(len(data_pret) * 0.85)

# 데이터 분할
train = data_pret[:train_end]
validation = data_pret[train_end:validation_end]
test = data_pret[validation_end:]

# Prophet 모델 학습
m = Prophet(yearly_seasonality=True, 
            weekly_seasonality=True, 
            daily_seasonality=True, 
            seasonality_mode='additive',
            )
m.fit(train)

# Cross-validation 수행
df_cv = cross_validation(m, 
                         initial='720 hours',
                        period='24 hours',
                        horizon='3 hours'
                         )

# 성능 지표 계산
df_p = performance_metrics(df_cv)
print(df_p[['horizon', 'rmse', 'mae']])

print("통계:")
print(df_p.describe())

# 모델 전체 RMSE 평균
rmse_mean = df_p['rmse'].mean()
print(f"RMSE Mean: {rmse_mean}")

# 테스트 데이터 예측
test_future = test['ds'].copy()
test_forecast = m.predict(test_future)

# 시각화
# 마지막 일주일
last_week_test = test_forecast[test_forecast['ds'] >= test_forecast['ds'].max() - pd.Timedelta(days=7)]

plt.figure(figsize=(12, 6))
plt.plot(last_week_test['ds'], test['y'].iloc[-len(last_week_test):], label='실제 Test값', color='black')
plt.plot(last_week_test['ds'], last_week_test['yhat'], label='예측 Test값', color='blue')
plt.fill_between(
    last_week_test['ds'], 
    last_week_test['yhat_lower'], 
    last_week_test['yhat_upper'], 
    color='blue', alpha=0.2, label='신뢰 구간'
)
plt.legend()
plt.title("1주일 도로 교통량")
plt.xlabel("Date")
plt.ylabel("Traffic Volume")
plt.show()

# # Prophet 모델 저장
# print("모델저장 시작")
# import joblib
# joblib.dump(m, 'prophet_no_variable_model.pkl')
# print("모델이 'prophet_no_variable_model.pkl'로 저장되었습니다.")
