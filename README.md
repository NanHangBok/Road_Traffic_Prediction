# Road_Traffic_Prediction
빅데이터처리 기말대체 평가<br>
도로 교통량 예측
<hr>

## 목차
- [1. 데이터 소개](#1-데이터-소개)
- [2. 데이터 처리](#2-데이터-처리)
- [3. XGBoost](#3-XGBoost)
- [4. Prophet](#4-Prophet)
- [5. NeuralProphet](#5-NeuralProphet)
  
<hr>

## 1. 데이터 소개
### 인천광역시 교통정보센터 제공 API 활용 <br>
- 도로 교통량 정보 제공
### 기상청 제공 초단기실황 API 활용 <br>
- 최대 24시간 전 정보 제공
- 최근 날씨 정보 제공
### 기상청 제공 지상 시간자료 조회서비스 API 활용 <br>
- 전일(D-1)까지 정보 제공
- 과거 날씨 정보 제공

<hr>

## 2. 데이터 처리
`start_day = datetime(2023, 6, 20, 0, 0)
now = datetime.now()
end_day = now - timedelta(hours=23)`<br>
데이터는 2023년 6월 20일 부터 추출하였고 그 이전에 데이터에는 Missing 데이터가 다수 존재하였다.<br>
시계열 데이터를 처리할 때는 연속성이 중요하다.<br>
### 데이터 수집
- 과거 날씨 정보
  - 날짜와 기온, 강수량, 풍속을 가져온다.
- 최근 날씨 정보
  - 마찬가지로 날짜와 기온, 강수량, 풍속을 가져온다. 
- 도로 교통량 정보
  - 하루치 데이터만 가져올 수 있기 때문에 하루씩 순회하며 데이터를 가져온다.
<br>

### 데이터 전처리
- 과거 날씨 정보
  - `historical_weather_df = historical_weather_df[['tm', 'ta', 'rn', 'ws']].rename(columns={'tm': 'datetime', 'ta': 'Temperature', 'rn': 'Rainfall', 'ws': 'WindSpeed'})`
  - api가 제공하는 형태가 tm, ta, rn, ws로 제공되기 때문에 Column명을 변경한다.<br>
  
- 최근 날씨 정보
  - `filtered_real_time_weather_df = real_time_weather_df[real_time_weather_df['category'].isin(['T1H', 'RN1', 'WSD'])]`
  - 제공되는 Category 중 T1H, RN1, WSD만 가져온다.
  - `real_time_weather_pivot = real_time_weather_pivot.rename(columns={'T1H': 'Temperature', 'RN1': 'Rainfall', 'WSD': 'WindSpeed'})`
  - 해당하는 column 명을 각 각 알맞게 수정한다. ('T1H': 'Temperature', 'RN1': 'Rainfall', 'WSD': 'WindSpeed')<br>
  
- 도로 교통량 정보
  - `traffic_df = traffic_df[traffic_df['linkID'] == '1630021501'] # 데이터가 너무 많아서 한개의 도로만 사용`
  - ```
    traffic_melted_df = pd.melt(
     traffic_df, 
     id_vars=['statDate', 'roadName', 'linkID', 'direction', 'startName', 'endName'], 
     # 기존 데이터는 각 시간 별로 컬럼이 존재
     # Datetime 포맷을 통일하게 하기 위해 수정
     value_vars=[f'hour{str(i).zfill(2)}' for i in range(24)], 
     var_name='hour', 
     value_name='traffic_volume'
    )
    ```
  - 제공되는 정보가 날짜마다 같은 행에 각 시간 별 교통량이 제공되기 때문에 포맷을 통일하기 위해 수정
    | 날짜 | hour01 | hour 02|...|
    |---|:---:|:---:|:---:|
    |20240101|24|21|...|<br>
    
- 통합된 데이터
  - 날씨 데이터
    - `weather_df = pd.concat([historical_weather_df, real_time_weather_pivot]).drop_duplicates().sort_values(by='datetime')`
    - 과거 날씨 데이터와 최근 날씨 데이터 병합
    - `weather_df[['Temperature', 'Rainfall', 'WindSpeed']] = weather_df[['Temperature', 'Rainfall', 'WindSpeed']].fillna(0)`
    - 결측치 채우기<br>

  -`traffic_melted_df = traffic_melted_df[['datetime', 'traffic_volume']]`
  - 필요한 정보만 남기기
  -`merged_data = pd.merge(traffic_melted_df, weather_df, on='datetime', how='inner')`
<br>

### 데이터 변환
- Prophet
- `recent_data = merged_data[['datetime', 'traffic_volume', 'Temperature', 'Rainfall', 'WindSpeed']].rename(columns={'datetime': 'ds', 'traffic_volume': 'y'})`
- `recent_data = recent_data.sort_values(by='ds')`
- Prophet에서는 날짜를 ds, 결과를 y로 저장하여야 함


<hr>

## 3. XGBoost
### 결과
##### 최종 RMSE: 35.09832311812178

일주일<br>
![xgboost1](https://github.com/user-attachments/assets/c714472a-c306-4f21-9d14-cdeb14865d4c)

하루<br>
![xgboost2](https://github.com/user-attachments/assets/c34122cd-1792-4f12-b60f-73a46a5838e7)

예측<br>
![xgboost2](https://github.com/user-attachments/assets/d9208d6e-9994-4dd5-a035-c7fd15b78a00)

<hr>

## 4. Prophet

### 결과

#### 변수 O
##### RMSE Mean: 85.02706092595479
![prophet_1](https://github.com/user-attachments/assets/36ade19a-9889-44c2-b6d6-2efd78933c16)

#### 변수 없이 진행
##### RMSE Mean: 65.75078761356201
![prophet_no_variable_1](https://github.com/user-attachments/assets/9e528643-8ede-495b-a31c-581d75351889)

#### 새로운 정보 제공
![prophet_prediction_1](https://github.com/user-attachments/assets/ff4bccb7-6530-4520-90bb-7a4a159d0bdf)

<hr>

## 5. NeuralProphet

### 결과

#### 변수 O
##### RMSE:  45.29098892211914

한달<br>
![neural_prophet_1](https://github.com/user-attachments/assets/540a132f-cf74-4621-8d32-d318b19b1a0e)

일주일<br>
![neural_prophet_2](https://github.com/user-attachments/assets/8105660e-f2ab-4bbf-b501-738dea18ea28)

#### 변수 없이 진행
##### RMSE:  80.39910125732422
한달<br>
![neural_prophet_no_variable_1](https://github.com/user-attachments/assets/d1b29b11-a44d-42b3-8d37-9941c17f410c)

일주일<br>
![neural_prophet_no_variable_2](https://github.com/user-attachments/assets/90191a51-e5d6-4585-a700-f66346a724f4)

#### 새로운 정보 제공
![neural_prophet_prediction_1](https://github.com/user-attachments/assets/13c607bc-70e2-4d95-a136-f0705a2f99f1)
