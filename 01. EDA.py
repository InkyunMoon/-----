import os
import pandas as pd
# import pandas_profiling
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)

data_path = os.path.join(os.getcwd(), 'SCALE불량.csv')
raw_data = pd.read_csv(data_path, encoding='CP949')

# df 형태 확인
raw_data.shape # 720개의 데이터, 21개의 피쳐 확인

# 결측치와 데이터 타입
raw_data.info() # 모두 720개로 결측치 없음 확인, 8개의 범주형 데이터(object)와 float64, int64 타입으로 구성

raw_data['FUR_NO_ROW'] = raw_data['FUR_NO_ROW'].astype('object')
raw_data['WORK_GR'] = raw_data['WORK_GR'].astype('object')
raw_data['HSB'] = raw_data['HSB'].astype('object')
raw_data['SCALE'] = raw_data['SCALE'].astype('object')

# raw_data.drop('ROLLING_DATE', axis=1, inplace = True) # 날짜 제거

############################## 1. 먼저 범주형 데이터를 살펴본다.##############################
raw_data.select_dtypes(include='object').info() # PLATE_NO ROLLING_DATE SCALE SPEC STEEL_KIND FUR_NO HSB WORK_GR 에 대해서 각각 조사

for col in raw_data.select_dtypes(include='object'):
    print('##########',col,'##########')
    print(raw_data[col].value_counts()) # ROLLING_DATE -> 너무 세세한 시간별로 나뉨, 시간대별로 범주화하기 -> 아직 완료X
    
# plate_no
sum(raw_data['PLATE_NO'].value_counts().values == 1) # 데이터별로 유니크한 값을 가지므로 제거
raw_data.drop('PLATE_NO', axis=1, inplace = True)

# SCALE
raw_data['SCALE'] = np.where(raw_data['SCALE'] == '양품', 0, 1) # 불량:1 양품:0으로 변환
print('불량품이 전체의' ,sum(raw_data['SCALE'] == 1)/(raw_data['SCALE']).count(),'%를 차지')
sns.countplot(raw_data['SCALE'])

# SPEC --> 보류
raw_data['SPEC'].value_counts().plot(kind='bar')
len(raw_data['SPEC'].unique()) # 66개 종류의 SPEC
len(raw_data["SPEC"].value_counts()[raw_data["SPEC"].value_counts() == 1]) # 유니크한 스펙의 개수 == 12
idx_more4 = raw_data["SPEC"].value_counts()[raw_data["SPEC"].value_counts() >= 4].index
spec4_data = []
#가설 검정을 통해 삭제할지말지 결정하자... 전에 선형결합으로 나타낼 수 있다는 정보는 어디서 얻었는지???

# STEEL_KIND
raw_data['STEEL_KIND'].value_counts() # C0 강재가 대다수(69.86%)를 차지함을 알 수 있다.
sns.countplot('STEEL_KIND', data = raw_data, hue = 'SCALE') # CO강재에서 스케일이 크게 발생한다.
STEEL_KIND_cross = pd.crosstab(raw_data.STEEL_KIND,raw_data.SCALE)

STEEL_KIND_cross[1]/STEEL_KIND_cross.sum(axis= 1) # CO강종에서 42%의 불량률이 발생

# FUR_NO @@가설검정: 화로에 따라 스케일 발생률에 차이가 있는가?
raw_data['FUR_NO'].value_counts() # 세개의 가열로 호기가 균등하게 분포
sns.catplot(x='FUR_NO', y='SCALE', data = raw_data, kind='bar', height = 4, palette = 'muted')

# FUR_NO_ROW
raw_data['FUR_NO_ROW'].value_counts() # 2개의 가열로가 균등하게 분포...?
sns.catplot(x='FUR_NO_ROW', y='SCALE', data = raw_data, kind='bar', height = 4, palette = 'muted')

# HSB
raw_data['HSB'] = np.where(raw_data['HSB'] == '적용', 1, 0) # 미적용:1 적용:0으로 변환
raw_data['HSB'].value_counts() # 적용된 경우가 대다수
sns.catplot(x='HSB', y = 'SCALE', data = raw_data, height = 4, kind = 'bar', palette = 'muted') # HSB 미적용시 스케일 100%. 적용시 30%수준으로 낮출 수 있음

# WORK_GR @@가설검정: 조에 따라 스케일 발생률에 차이가 있는가?
raw_data['WORK_GR'] = np.where(raw_data['WORK_GR'] == '1조', 1, np.where(raw_data['WORK_GR'] == '2조', 2, np.where(raw_data['WORK_GR'] == '3조', 3, 4))) # 조 번호만 남긴다.
raw_data['WORK_GR'].value_counts() # 균등하게 분배됨
sns.catplot(x='WORK_GR', y='SCALE', data = raw_data, kind='bar', height = 4, palette = 'muted') # 2조가 가장 스케일 발생이 낮음

############################## 2. 연속형 데이터를 살펴본다.##############################
''' 
[PT_THK', 'PT_WDTH', 'PT_LTH', 'PT_WGT', 'FUR_NO_ROW',
       'FUR_HZ_TEMP', 'FUR_HZ_TIME', 'FUR_SZ_TEMP', 'FUR_SZ_TIME', 'FUR_TIME',
       'FUR_EXTEMP', 'ROLLING_TEMP_T5', 'HSB', 'ROLLING_DESCALING', 'WORK_GR']
'''
data_cont = raw_data.select_dtypes(exclude='object')
data_cont.drop(['HSB','WORK_GR'], axis=1, inplace = True)
data_cont.describe() # ROLLING_TEMP_T5에 0값이 있음을 확인 -> 나중에 처리

# 전체적인 히트맵 그려보기
fig, ax = plt.subplots( figsize=(22,12) )
corr = data_cont.corr()

mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(corr, 
            cmap = 'RdYlBu_r', 
            annot = True,   # 실제 값을 표시한다
            mask=mask,      # 표시하지 않을 마스크 부분을 지정한다
            linewidths=.5,  # 경계면 실선으로 구분하기
            cbar_kws={"shrink": .5},# 컬러바 크기 절반으로 줄이기
            annot_kws={"size": 20},
            vmin = -1,vmax = 1)   # 컬러바 범위 -1 ~ 1  

# FUR_SZ_TEMP와 FUR_EXTREMP의 상관관계가 1. FUR_EXTEMP 열 드랍(가열로 균열대 온도와 가열로에서 나왔을때의 온도가 당연히 비례할 것)
raw_data.drop('FUR_EXTEMP', axis=1, inplace = True)
data_cont.drop('FUR_EXTEMP', axis=1, inplace = True)

# FUR_SZ_TEMP와 FUR_HZ_TEMP -> 0.77
sns.pairplot(data_cont)
##### -> 공선성문제, 회귀 모델에서 해결하도록 한다.
# 산점도를 통해 ROLLING_DESCALING과 SCALE간에 일정한 패턴이 존재함을 확인
# ROLLING_DESCALING
sns.catplot(x='ROLLING_DESCALING', y='SCALE', data=raw_data) # DESCALING이 홀수인 경우, 스케일링이 100%발생한다.
# 산점도 -> FUR_SZ_TEMP가 1175를 초과하면 반드시 스케일링 발생
# 산점도 -> ROLLING_TEMP_T5가 1000을 초과하면 반드시 스케일링 발생
# 산점도 -> FUR_HZ_TEMP가 1188를 초과하면 스케일링 발생빈도 급격히 증가

# ROLLING_TEMP_T5
sns.catplot(y = 'ROLLING_TEMP_T5', data = raw_data, height = 4, kind = 'box', palette = 'muted') # 0이 6개 있다.
raw_data[raw_data['ROLLING_TEMP_T5'] == 0] # 모두 스케일이 발생하지 않았으며, 특정 시간대에 겹쳐서 등장하는 경향이 있으므로 측정 센서의 오류라고 판단, 평균치로 대체하도록 하자.

mean_t5 = int(round(raw_data['ROLLING_TEMP_T5'].mean(),0))
idx_zero_t5 = raw_data[raw_data['ROLLING_TEMP_T5'] == 0]['ROLLING_TEMP_T5'].index

raw_data['ROLLING_TEMP_T5'] = raw_data['ROLLING_TEMP_T5'].apply(lambda x: mean_t5 if x == 0 else x)

# 전체적인 박스플랏 그려보기
for col in data_cont.columns:
    sns.catplot(y = col, data = raw_data, kind = 'box')

# PT_THK
# PT_WDTH
# FUR_HZ_TIME
# FUR_SZ_TIME 이 네가지, 1.5*IQR을 벗어나는 데이터 다수 존재. 4개 피쳐에서 동시에 2개 이상의 피쳐가 fence를 벗어나는 경우, 삭제하도록 하자.

# 데이터프레임, 살펴볼 피쳐, 개수를 입력하면 이상치의 인덱스를 리턴하는 함수를 만든다.
def detect_outliers(df, n, features):
    outlier_indices = []
    
    for col in features:
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IQR = Q3 - Q1
        
        outlier_step = 1.5*IQR
        
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        # outlier_list_col 은 조건에 맞는 인덱스 번호 리턴
        
        outlier_indices.extend(outlier_list_col)
        
    outlier_indices = Counter(outlier_indices) # 인덱스: 개수 형태로 저장
    multiple_outliers = list(k for k, v in outlier_indices.items() if v>=n)
    # 탐지된 이상치가 n개 이상인 샘플에 대한 인덱스 리스트를 만든다.
    return multiple_outliers

# 두개 이상 피쳐에서 아웃라이어가 있는 인덱스 제거?
idx_selected = detect_outliers(raw_data, 2, ['PT_THK','PT_WDTH','FUR_HZ_TIME','FUR_SZ_TIME'])
# 1개 이상 피쳐로 선택하면 114개의 데이터가 선택되므로 보수적 조건(2개) 설정
raw_data.drop(idx_selected, axis=0, inplace = True)
# -> 선택된 데이터 모두 스케일이 발생하지 않음. 그러나 평균 두 피쳐에서 평균으로부터 크게 벗어나 있으므로 제거한다.

# 논문 참고, 파생변수 생성
raw_data['temp_loss'] = raw_data['FUR_SZ_TIME'] - raw_data['ROLLING_TEMP_T5']
# -> loss가 크면 스케일링 발생빈도 증가



