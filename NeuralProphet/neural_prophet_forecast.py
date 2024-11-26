"""
Neural Prophet 예측
"""

import pandas as pd
import torch
import matplotlib.pyplot as plt
from neuralprophet import NeuralProphet

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 새로운 파일 불러오기
# 학습에 사용된 데이터와 다른 데이터
data = pd.read_csv('neuralprophet_input_data.csv')
# print("입력 데이터:")
# print(data)

# 저장된 모델 로드
model_path = 'neuralprophet_model.pt'
model = torch.load(model_path)

# 예측 수행
forecast = model.predict(data)

# 예측 결과에 입력 데이터 포함
forecast_with_input = pd.concat([data, forecast[['ds', 'yhat1', 'yhat2', 'yhat3']]], axis=1)

# 결과 확인
print("예측 결과:")
print(forecast_with_input)

# 시각화
# neuralProphet은 각 시간별로 예측값을 따로 저장함
# yhat1 = 1시간 , yhat2 = 2시간 뒤 , yhat3 = 3시간 뒤
plt.figure(figsize=(14, 6))
plt.plot(forecast['ds'], forecast['y'], label="실제 데이터", color="blue", marker="o")  # 입력 데이터
plt.plot(forecast['ds'], forecast['yhat1'], label="예측 1시간", color="orange", linestyle="--", marker="x")  # 예측값
plt.plot(forecast['ds'], forecast['yhat2'], label="예측 2시간", color="green", linestyle="--", marker="x")  # 2시간 예측
plt.plot(forecast['ds'], forecast['yhat3'], label="예측 3시간", color="red", linestyle="--", marker="x")  # 3시간 예측
plt.title("3시간 예측")
plt.xlabel("시간")
plt.ylabel("교통량")
plt.legend()
plt.grid()
plt.show()
