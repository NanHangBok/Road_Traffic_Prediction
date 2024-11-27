import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import numpy as np
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv('./api.env')

# API 키 및 URL 설정
real_time_weather_api_key = os.getenv('real_time_weather_api_key')
traffic_api_key = os.getenv('traffic_api_key')
past_weather_url = os.getenv('real_time_weather_url')
future_weather_url = os.getenv('future_weather_url')
traffic_url = os.getenv('traffic_url')

# 최근 발표 시간을 계산하는 함수
def get_recent_base_time(now):
    base_times = ["0230", "0530", "0830", "1130", "1430", "1730", "2030", "2330"]
    now_str = now.strftime("%H%M")
    for base_time in reversed(base_times):
        if now_str >= base_time:
            return base_time
    return base_times[-1]

# 현재 시각 기준으로 발표 시간 설정
now = datetime.now()
base_time = get_recent_base_time(now)
base_date = now.strftime("%Y%m%d") if base_time != "2330" else (now - timedelta(days=1)).strftime("%Y%m%d")
print(f"Base Date: {base_date}, Base Time: {base_time}")

# 과거 데이터 수집
past_weather_data = []
for hour_offset in range(1, 24):  # 과거 23시간 데이터 수집
    target_time = now - timedelta(hours=hour_offset)
    target_time = target_time.replace(minute=0, second=0, microsecond=0)
    date_str = target_time.strftime('%Y%m%d')
    time_str = target_time.strftime('%H%M')
    params = {
        'serviceKey': real_time_weather_api_key,
        'numOfRows': 1000,
        'pageNo': 1,
        'dataType': 'JSON',
        'base_date': date_str,
        'base_time': time_str,
        'nx': 54,
        'ny': 124,
        
    }

    try:
        response = requests.get(past_weather_url, params=params)
        if response.status_code == 200:
            data = response.json()
            items = data.get('response', {}).get('body', {}).get('items', {}).get('item', [])
            for item in items:
                if 'category' in item and 'obsrValue' in item:
                    past_weather_data.append({
                        'category': item['category'],
                        'value': float(item['obsrValue']),
                        'datetime': target_time
                    })
    except Exception as e:
        print(f"과거 데이터 요청 실패: {e}")

past_weather_df = pd.DataFrame(past_weather_data)

# 필요한 카테고리 필터링 (T1H: 기온, RN1: 강수량, WSD: 풍속)
if not past_weather_df.empty:
    past_weather_df = past_weather_df[past_weather_df['category'].isin(['T1H', 'RN1', 'WSD'])]
    past_weather_pivot = past_weather_df.pivot_table(
        index='datetime', columns='category', values='value'
    ).reset_index()
    past_weather_pivot = past_weather_pivot.rename(
        columns={'T1H': 'Temperature', 'RN1': 'Rainfall', 'WSD': 'WindSpeed'}
    )
else:
    past_weather_pivot = pd.DataFrame()

# 미래 데이터 수집
future_weather_data = []
params = {
    'serviceKey': real_time_weather_api_key,
    'numOfRows': 100,
    'pageNo': 1,
    'dataType': 'JSON',
    'base_date': base_date,
    'base_time': base_time,
    'nx': 54,
    'ny': 124
}

try:
    response = requests.get(future_weather_url, params=params)
    if response.status_code == 200:
        data = response.json()
        items = data.get('response', {}).get('body', {}).get('items', {}).get('item', [])
        for item in items:
            if 'category' in item and 'fcstValue' in item:
                fcst_value = item['fcstValue']
                if isinstance(fcst_value, str) and fcst_value == "강수없음":
                    fcst_value = 0.0
                else:
                    try:
                        fcst_value = float(fcst_value)
                    except ValueError:
                        continue
                future_weather_data.append({
                    'category': item['category'],
                    'value': fcst_value,
                    'datetime': pd.to_datetime(item['fcstDate'] + item['fcstTime'], format='%Y%m%d%H%M')
                })
except Exception as e:
    print(f"미래 데이터 요청 실패: {e}")

future_weather_df = pd.DataFrame(future_weather_data)

# 필요한 카테고리 필터링 (T1H: 기온, RN1: 강수량, WSD: 풍속)
if not future_weather_df.empty:
    future_weather_df = future_weather_df[future_weather_df['category'].isin(['T1H', 'RN1', 'WSD'])]
    future_weather_pivot = future_weather_df.pivot_table(
        index='datetime', columns='category', values='value'
    ).reset_index()
    future_weather_pivot = future_weather_pivot.rename(
        columns={'T1H': 'Temperature', 'RN1': 'Rainfall', 'WSD': 'WindSpeed'}
    )
else:
    future_weather_pivot = pd.DataFrame()

# 현재 시간 데이터 확인 및 채우기
if not past_weather_pivot.empty and now.replace(minute=0, second=0, microsecond=0) not in past_weather_pivot['datetime'].values:
    current_time_row = future_weather_pivot[future_weather_pivot['datetime'] == now.replace(minute=0, second=0, microsecond=0)]
    if not current_time_row.empty:
        past_weather_pivot = pd.concat([past_weather_pivot, current_time_row])

past_weather_pivot.to_csv('te1.csv', index=False, encoding='utf-8-sig')

# 과거 데이터와 미래 데이터 병합
final_weather_data = pd.concat([past_weather_pivot, future_weather_pivot]).drop_duplicates(subset='datetime').sort_values(by='datetime').reset_index(drop=True)

# 현재 시간 기준으로 전날 02시부터 1시간 전까지의 예측 범위 설정
now = datetime.now().replace(minute=0, second=0, microsecond=0)
end_date = now - timedelta(hours=1)  # 현재 시간에서 1시간 전까지만 설정
start_time = end_date - timedelta(days=7)  # 1주일 전부터 데이터 수집

# 교통량 데이터 수집
print('교통량 수집 시작')
traffic_data = []
current_date = start_time.date()

while current_date <= end_date.date():
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

print("모든 데이터 수집 완료.")

# 데이터 프레임 생성
traffic_df = pd.DataFrame(traffic_data)
traffic_df = traffic_df[traffic_df['linkID'] == '1630021501']
traffic_melted_df = pd.melt(
    traffic_df, 
    id_vars=['statDate', 'roadName', 'linkID', 'direction', 'startName', 'endName'], 
    value_vars=[f'hour{str(i).zfill(2)}' for i in range(24)], 
    var_name='hour', 
    value_name='traffic_volume'
)
traffic_melted_df['hour'] = traffic_melted_df['hour'].apply(lambda x: x.replace('hour', '') + ":00")
traffic_melted_df['datetime'] = pd.to_datetime(traffic_melted_df['statDate'] + ' ' + traffic_melted_df['hour'], format='%Y-%m-%d %H:%M')
traffic_melted_df = traffic_melted_df[['datetime', 'traffic_volume']]
traffic_melted_df = traffic_melted_df[traffic_melted_df['datetime'] <= end_date]

# 데이터 병합
final_data = pd.merge(final_weather_data, traffic_melted_df, on='datetime', how='inner')

# 최종 데이터 확인
print(final_data)

# 현재 시간 반올림 처리
now_rounded = now.replace(minute=0, second=0, microsecond=0)

# 현재 시간을 포함한 3시간 미래 데이터 필터링
future_3hours_data = future_weather_pivot[
    (future_weather_pivot['datetime'] >= now_rounded)
].head(3)

# traffic_volume 컬럼 추가 및 NaN으로 채우기
if not future_3hours_data.empty:
    future_3hours_data['traffic_volume'] = np.nan
    # 최종 데이터 병합
    extended_final_data = pd.concat([final_data, future_3hours_data], ignore_index=True).sort_values(by='datetime')
else:
    print("3시간 미래 데이터가 없습니다.")
    extended_final_data = final_data.copy()

# day_of_week와 hour 추가
extended_final_data['day_of_week'] = extended_final_data['datetime'].dt.dayofweek  # 요일 (0=월요일, 6=일요일)
extended_final_data['hour'] = extended_final_data['datetime'].dt.hour             # 시간 (0~23)

# NeuralProphet 입력 데이터 형식으로 변환
neuralprophet_input_data = extended_final_data.rename(
    columns={
        "datetime": "ds",
        "traffic_volume": "y",
        "Temperature": "Temperature",
        "Rainfall": "Rainfall",
        "WindSpeed": "WindSpeed",
        "day_of_week": "day_of_week",
        "hour": "hour"
    }
)

# NeuralProphet용 컬럼만 선택
neuralprophet_input_data = neuralprophet_input_data[['ds', 'y', 'Temperature', 'Rainfall', 'WindSpeed', 'day_of_week', 'hour']]

# CSV 저장
# neuralprophet_input_data.to_csv('neuralprophet_input_data123.csv', index=False, encoding='utf-8-sig')

