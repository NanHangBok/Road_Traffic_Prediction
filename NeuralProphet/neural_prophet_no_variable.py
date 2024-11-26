"""
NeuralProphet
No Variable (변수없음)
"""

import pandas as pd
import matplotlib.pyplot as plt
from neuralprophet import NeuralProphet
import plotly.express as px

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 불러오기
# 매번 데이터를 모아서 사용하기 오래걸려서 여기서는 csv로 저장 후 불러와 사용
recent_data = pd.read_csv('4.csv', parse_dates=['ds'])

# NeuralProphet 형식의 데이터 준비
data_pret = recent_data[['ds', 'y']].copy()  # 필요한 컬럼만 선택
data_pret = data_pret.set_index('ds')
data_pret = data_pret.asfreq('H')  # 시간 간격을 시간 단위로 설정
data_pret = data_pret.ffill()  # NaN 값을 이전 값으로 채움
data_pret = data_pret.reset_index()

# 데이터 분할
# (70% Train, 15% Validation, 15% Test)
train_end = int(len(data_pret) * 0.7)
validation_end = int(len(data_pret) * 0.85)

train = data_pret[:train_end]
valid = data_pret[train_end:validation_end]
test = data_pret[validation_end:]

# NeuralProphet 모델 설정
m = NeuralProphet(
    growth='off',                  # 추세 유형 설정
    yearly_seasonality=True,       # 연간 계절성 설정
    weekly_seasonality=True,       # 주간 계절성 설정
    daily_seasonality=True,        # 일간 계절성 설정
    batch_size=64,                 # 배치 사이즈 설정
    epochs=200,                    # 학습 횟수 설정
    learning_rate=0.01,            # 학습률 설정
    n_lags=24,                     # 과거 데이터 24시간 참조
    n_forecasts=3                 # 3시간 예측
)

# 공휴일 및 계절성 추가
m.add_country_holidays("KR")  # 한국 공휴일 추가
m.add_seasonality(name="monthly", period=30.5, fourier_order=3) # 월간 계절성 설정

# 학습 수행
metrics = m.fit(train, freq='h', validation_df=valid, progress='plot')

# 평가 지표 출력
print("RMSE: ", metrics.RMSE.tail(1).item())
print("MAE(Train): ", metrics.MAE.tail(1).item())
print("MAE(Test): ", metrics.MAE_val.tail(1).item())

# 성능 변화 시각화
px.line(metrics, y=['MAE', 'MAE_val'], width=800, height=400)

#yhat1과 실제값 시각화(lag 데이터 포함x)
forecast = m.predict(test)
fig = m.plot(forecast[['ds', 'y', 'yhat1']])

# 마지막 한 달 시각화
last_month = forecast[forecast['ds'] >= forecast['ds'].max() - pd.Timedelta(days=30)]
print(last_month['y'].tail(30))
print()
print(last_month['yhat1'].tail(30))
plt.figure(figsize=(12, 6))
plt.plot(last_month['ds'], last_month['y'], label='실제 값', color='black')
plt.plot(last_month['ds'], last_month['yhat1'], label='예측 값', color='blue')
plt.legend()
plt.title("한달 도로 교통량")
plt.xlabel("Date")
plt.ylabel("Traffic Volume")
plt.show()

# 일주일 시각화
# 마지막 이주일 데이터를 가져오고, 그 중 마지막 7일 필터링
# 마지막 1주 데이터의 미싱값이 존재

end_date = forecast['ds'].max() - pd.Timedelta(days=7)  # 일주일 전
start_date = forecast['ds'].max() - pd.Timedelta(days=14)  # 이주일 전

last_week = forecast[(forecast['ds'] >= start_date) & (forecast['ds'] < end_date)]

plt.figure(figsize=(12, 6))
plt.plot(last_week['ds'], last_week['y'], label='실제 값', color='black')
plt.plot(last_week['ds'], last_week['yhat1'], label='예측 값', color='blue')  # yhat으로 변경
plt.legend()
plt.title("1주일 도로 교통량 (데이터 기준 2주일 전)")
plt.xlabel("Date")
plt.ylabel("Traffic Volume")
plt.show()
