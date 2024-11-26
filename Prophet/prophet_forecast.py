"""
Prophet 예측
"""

import pandas as pd
import joblib
import matplotlib.pyplot as plt
from prophet import Prophet

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 저장된 Prophet 모델 로드
model = joblib.load('prophet_model.pkl')
print("Prophet 모델이 로드되었습니다.")

# 새로운 데이터 불러오기
# 학습에 사용된 데이터와 다른 데이터
data = pd.read_csv('neuralprophet_input_data.csv', parse_dates=['ds'])
print("입력 데이터:")
print(data)

# 데이터 확인
if 'y' in data.columns:
    print("\n현재 데이터셋에 y 값 포함.")
else:
    print("\n현재 데이터셋에 y 값이 없습니다. NaN 값이 예측 대상이 될 것입니다.")

# Prophet 모델로 예측 수행
forecast = model.predict(data)

# 예측 결과에 입력 데이터 포함
forecast_with_input = pd.concat([data, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]], axis=1)

# 결과 확인
print("예측 결과:")
print(forecast_with_input)

# 시각화
plt.figure(figsize=(14, 6))
plt.plot(data['ds'], data['y'], label="실제 데이터", color="blue", marker="o")  # 입력 데이터
plt.plot(forecast['ds'], forecast['yhat'], label="예측값", color="orange", linestyle="--")  # 예측값
plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color="orange", alpha=0.2, label="예측 신뢰 구간") # 예측 신뢰구간
plt.title("예측 결과")
plt.xlabel("시간")
plt.ylabel("교통량")
plt.legend()
plt.grid()
plt.show()
