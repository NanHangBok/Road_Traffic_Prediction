import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from prophet import Prophet
import time
from neuralprophet import NeuralProphet, set_log_level
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv('./api.env')

# API 설정
historical_weather_api_key = os.getenv('historical_weather_api_key')
real_time_weather_api_key = os.getenv('real_time_weather_api_key')
traffic_api_key = os.getenv('traffic_api_key')
historical_weather_url = os.getenv('historical_weather_url')
real_time_weather_url = os.getenv('real_time_weather_url')
traffic_url = os.getenv('traffic_url')

# 2023년 6월 20일 00시로 시작 날짜 설정
# 이전 데이터는 중간 중간 Missing 값이 많음
start_day = datetime(2023, 6, 20, 0, 0)
now = datetime.now()
end_day = now - timedelta(hours=23)

start_day_str = start_day.strftime('%Y%m%d')
end_day_str = end_day.strftime('%Y%m%d')
end_hour_str = end_day.strftime('%H')

print('과거 날씨 데이터 수집 시작')

# 수집할 전체 데이터 리스트 초기화
historical_weather_data = []

# API 요청 파라미터 설정
params = {
    'serviceKey': historical_weather_api_key,
    'pageNo': 1,
    'numOfRows': 999,
    'dataType': 'JSON',
    'dataCd': "ASOS",
    'dateCd': "HR",
    'startDt': start_day_str, # (YYYYMMDD)
    'startHh': '00', # (HH)
    'endDt': end_day_str, # (YYYYMMDD) (전일(D-1) 까지 제공)
    'endHh': end_hour_str, # (HH)
    'stnIds': 112 # 인천 수도권기상청
}

response = requests.get(historical_weather_url, params=params)
if response.status_code == 200:
    data = response.json()
    items = data['response']['body']['items']['item']
    historical_weather_data.extend(items)

historical_weather_df = pd.DataFrame(historical_weather_data)
historical_weather_df = historical_weather_df[['tm', 'ta', 'rn', 'ws']].rename(columns={'tm': 'datetime', 'ta': 'Temperature', 'rn': 'Rainfall', 'ws': 'WindSpeed'})
historical_weather_df['datetime'] = pd.to_datetime(historical_weather_df['datetime'])
print(historical_weather_df)

# 실시간 날씨 데이터 수집
# api로 얻을 수 있는 과거 데이터가 전일 까지 제공하기 때문에
print('실시간 날씨 데이터 수집 시작')
real_time_weather_data = []
for hour_offset in range(23):  # 최근 23시간 데이터 수집
    target_time = now - timedelta(hours=hour_offset)
    date_str = target_time.strftime('%Y%m%d')
    time_str = target_time.strftime('%H00')

    params = {
        'serviceKey': real_time_weather_api_key,
        'pageNo': 1,
        'numOfRows': 1000,
        'dataType': 'JSON',
        'base_date': date_str,
        'base_time': time_str,
        'nx': 54,  # 학익 1동
        'ny': 124  # 학익 1동
    }

    try:
        response = requests.get(real_time_weather_url, params=params)
        if response.status_code == 200:
            data = response.json()
            items = data.get('response', {}).get('body', {}).get('items', {}).get('item', [])
            for item in items:
                if 'obsrValue' in item and 'category' in item:
                    item['obsrValue'] = pd.to_numeric(item['obsrValue'], errors='coerce')
                    item['baseDate'] = date_str
                    item['baseTime'] = time_str
                    real_time_weather_data.append(item)
    except requests.exceptions.RequestException as e:
        print(f"실시간 날씨 데이터 요청 실패: {e}")

real_time_weather_df = pd.DataFrame(real_time_weather_data)

# T1H = 온도 , RN1 = 강수량 , WSD = 풍속
filtered_real_time_weather_df = real_time_weather_df[real_time_weather_df['category'].isin(['T1H', 'RN1', 'WSD'])]

if not filtered_real_time_weather_df.empty:
    real_time_weather_pivot = filtered_real_time_weather_df.pivot_table(
        index=['baseDate', 'baseTime'], columns='category', values='obsrValue'
    ).reset_index()
    real_time_weather_pivot.columns.name = None
    # T1H , RN1, WSD 의 컬럼 명 변경
    real_time_weather_pivot = real_time_weather_pivot.rename(columns={'T1H': 'Temperature', 'RN1': 'Rainfall', 'WSD': 'WindSpeed'})
    # Datetime 포맷 통일화
    real_time_weather_pivot['datetime'] = pd.to_datetime(
        real_time_weather_pivot['baseDate'] + real_time_weather_pivot['baseTime'].str.zfill(4), format='%Y%m%d%H%M'
    )
    real_time_weather_pivot = real_time_weather_pivot[['datetime', 'Temperature', 'Rainfall', 'WindSpeed']]
else:
    print("실시간 날씨 데이터가 비어 있습니다.")

print(real_time_weather_df)
# 두 개의 날씨 데이터 병합 및 결측값 처리
weather_df = pd.concat([historical_weather_df, real_time_weather_pivot]).drop_duplicates().sort_values(by='datetime')
# 결측치 채우기
weather_df[['Temperature', 'Rainfall', 'WindSpeed']] = weather_df[['Temperature', 'Rainfall', 'WindSpeed']].fillna(0)
# print(weather_df)
# weather_df.to_csv('1.csv',index=False, encoding='utf-8-sig')

"""
*
* 교통량 수집
*
"""
print('교통량 수집 시작')
traffic_data = []
current_date = datetime.strptime(start_day_str, '%Y%m%d')

# 날짜 순회
while current_date <= datetime.strptime(end_day_str, '%Y%m%d'):
    date_str = current_date.strftime('%Y%m%d')
    page_no = 1
    daily_data = []
 
    while True:
        params = {
            'serviceKey': traffic_api_key,
            'pageNo': page_no,
            'numOfRows': 2000,
            'YMD': date_str
        }

        try:
            response = requests.get(traffic_url, params=params)
            if response.status_code == 200:
                data = response.json()
                items = data['response']['body']['items']

                if items:
                    daily_data.extend(items)
                if len(items) < 2000:
                    break
                page_no += 1
            else:
                break
        except requests.exceptions.RequestException as e:
            time.sleep(5)

    current_date += timedelta(days=1)
    traffic_data.extend(daily_data)

traffic_df = pd.DataFrame(traffic_data)
traffic_df = traffic_df[traffic_df['linkID'] == '1630021501'] # 데이터가 너무 많아서 한개의 도로만 사용
traffic_melted_df = pd.melt(
    traffic_df, 
    id_vars=['statDate', 'roadName', 'linkID', 'direction', 'startName', 'endName'], 
    # 기존 데이터는 각 시간 별로 컬럼이 존재
    # Datetime 포맷을 통일하게 하기 위해 수정
    value_vars=[f'hour{str(i).zfill(2)}' for i in range(24)], 
    var_name='hour', 
    value_name='traffic_volume'
)
traffic_melted_df['hour'] = traffic_melted_df['hour'].apply(lambda x: x.replace('hour', '') + ":00") # HH:00 포맷
traffic_melted_df['datetime'] = pd.to_datetime(traffic_melted_df['statDate'] + ' ' + traffic_melted_df['hour'], format='%Y-%m-%d %H:%M')
traffic_melted_df = traffic_melted_df[['datetime', 'traffic_volume']]
#traffic_melted_df.to_csv('2.csv',index=False, encoding='utf-8-sig')

# 데이터 병합 및 결측값 처리
merged_data = pd.merge(traffic_melted_df, weather_df, on='datetime', how='inner')
merged_data['traffic_volume'] = merged_data['traffic_volume'].astype(int)
merged_data = merged_data.sort_values(by='datetime')

# Temperature, Rainfall, WindSpeed 열에서 숫자가 아닌 값을 NaN으로 변환한 후 0으로 채우기
for column in ['Temperature', 'Rainfall', 'WindSpeed']:
    merged_data[column] = pd.to_numeric(merged_data[column], errors='coerce').fillna(0)

# 추후 수정 코드 데이터프레임 준비
# PROPHET에서는 ds와 y로 사용되어야 함
recent_data = merged_data[['datetime', 'traffic_volume', 'Temperature', 'Rainfall', 'WindSpeed']].rename(columns={'datetime': 'ds', 'traffic_volume': 'y'})
recent_data = recent_data.sort_values(by='ds')
#recent_data.to_csv('4.csv',index=False, encoding='utf-8-sig')