import os
import pandas as pd
# import pandas_profiling
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import f_oneway, ttest_ind

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
raw_data['SCALE'] = np.where(raw_data['SCALE'] == '양품', 0, 1) # 불량:1 양품:0으로 변환
raw_data['SCALE'] = raw_data['SCALE'].astype('object')

raw_data.drop('ROLLING_DATE', axis=1, inplace = True) # 날짜 제거

############################## 1. 먼저 범주형 데이터를 살펴본다.##############################
raw_data.select_dtypes(include='object').info() # PLATE_NO SCALE SPEC STEEL_KIND FUR_NO HSB WORK_GR 에 대해서 각각 조사

for col in raw_data.select_dtypes(include='object'):
    print('##########',col,'##########')
    print(raw_data[col].value_counts()) 

# SCALE
print('불량품이 전체의', sum(raw_data['SCALE'] == 1)/(raw_data['SCALE']).count(),'%를 차지')
sns.countplot(raw_data['SCALE'])

# plate_no
sum(raw_data['PLATE_NO'].value_counts().values == 1) # 데이터별로 유니크한 값을 가지므로 제거
raw_data.drop('PLATE_NO', axis=1, inplace = True)

# SPEC --> 보류

plt.figure(figsize=(20,12))
# raw_data['SPEC'].value_counts().plot(kind='bar')
g = sns.countplot(x='SPEC', data=raw_data, hue='SCALE', orient='h')
g.set_xticklabels(g.get_xticklabels(), 
                          rotation=90, 
                          horizontalalignment='right')
plt.legend(title='SCALE', loc='upper right')
plt.setp(g.get_legend().get_texts(), fontsize='22') # for legend text
plt.setp(g.get_legend().get_title(), fontsize='32') # for legend title


########################스펙########################
# SPEC_KIND_cross = pd.crosstab(raw_data.SPEC, raw_data.SCALE)
SPEC_KIND_cross = pd.crosstab(raw_data.SPEC, raw_data.SCALE, margins=True)
SPEC_KIND_cross['scale_ratio'] = SPEC_KIND_cross.iloc[:,1]/(SPEC_KIND_cross['All'])
spec_high_scale = SPEC_KIND_cross[(SPEC_KIND_cross['scale_ratio'] >= 0.75) & (SPEC_KIND_cross['All'] >= 4)].index

raw_data['SPEC'].isin(spec_high_scale)

h_scale_idx = []
for idx, row in enumerate(raw_data['SPEC']):
    if row in spec_high_scale:
        h_scale_idx.append(idx)
        
for_test = raw_data.iloc[h_scale_idx]
sns.pairplot(for_test.select_dtypes(exclude='object'))

#--------------------- 반대 케이스 확인 - 차이 확인
spec_low_scale = SPEC_KIND_cross[(SPEC_KIND_cross['scale_ratio'] <= 0.25) & (SPEC_KIND_cross['All'] >= 4)].index

l_scale_idx = []
for idx, row in enumerate(raw_data['SPEC']):
    if row in spec_low_scale:
        l_scale_idx.append(idx)
        
for_test_low = raw_data.iloc[l_scale_idx]
sns.pairplot(for_test_low.select_dtypes(exclude='object'))

# SPEC 두 집단 모든 변수 히스토그램 그려보기

df_hscale_spec = raw_data[raw_data['SPEC'].isin(spec_high_scale)]
df_lscale_spec = raw_data[raw_data['SPEC'].isin(spec_low_scale)]


i = 0
fig = plt.figure(figsize = (15,10))
for row in range(2):    
    for col in list(raw_data.select_dtypes(exclude='object').columns):
        plt.subplot(row,6,i+1)
        i += 1
        plt.hist(df_hscale_spec[col], alpha=.5)
        plt.hist(df_lscale_spec[col], alpha=.5)
        plt.xlabel(col, fontsize = 15)
        plt.legend(fontsize=15)
plt.show()

rows = range(1,4)
cols = raw_data.select_dtypes(exclude='object').columns
nrows = 3
ncols = 4
fig, axes = plt.subplots(nrows,ncols, figsize = (20,18))
for row, a in enumerate(rows):
    for col, b in enumerate(cols):

        axes[row][col].hist(df_hscale_spec[b], alpha = 0.5)
        axes[row][col].hist(df_lscale_spec[b], alpha = 0.5)
        axes[row][col].set_xlabel(b)
        

# -> 특정 스펙에서 스케일 유난히 많이 발생함 확인
# --> 특정 스펙에서 온도 로스가 크게 나타남을 확인, 스펙을 특정 스펙(1)과 아닌 것(0)으로 구분

raw_data['specific_spec'] = 0
for idx, row in enumerate(raw_data['SPEC']):
    if row in spec_high_scale:
        raw_data.loc[idx,'specific_spec'] = 1

raw_data.drop('SPEC', axis=1, inplace = True)
# SPEC은 드랍해주도록 한다.

# STEEL_KIND
raw_data['STEEL_KIND'].value_counts() # C0 강재가 대다수(69.86%)를 차지함을 알 수 있다. 강재 종류 9개
sns.countplot('STEEL_KIND', data = raw_data, hue = 'SCALE') # CO강재에서 스케일이 크게 발생한다.
STEEL_KIND_cross = pd.crosstab(raw_data.STEEL_KIND,raw_data.SCALE)
STEEL_KIND_cross[1]/STEEL_KIND_cross.sum(axis= 1) # CO강종에서 42%의 불량률이 발생

# FUR_NO @@가설검정: 화로에 따라 스케일 발생률에 차이가 있는가?
raw_data['FUR_NO'].value_counts() # 세개의 가열로 호기가 균등하게 분포
raw_data['FUR_NO'] = raw_data['FUR_NO'].map(lambda x: str(x[0])) # '호기' 제거
sns.catplot(x='FUR_NO', y='SCALE', data = raw_data, kind='bar', height = 4, palette = 'muted')

# FUR_NO_ROW
raw_data['FUR_NO_ROW'].value_counts() # 2개의 가열로가 균등하게 분포...?
sns.catplot(x='FUR_NO_ROW', y='SCALE', data = raw_data, kind='bar', height = 4, palette = 'muted')

# HSB
raw_data['HSB'] = np.where(raw_data['HSB'] == '적용', str(1), str(0)) # 미적용:1 적용:0으로 변환 ######################
raw_data['HSB'].value_counts() # 적용된 경우가 대다수
sns.catplot(x='HSB', y = 'SCALE', data = raw_data, height = 4, kind = 'bar', palette = 'muted') # HSB 미적용시 스케일 100%. 적용시 30%수준으로 낮출 수 있음
sns.scatterplot(x='HSB', y='SCALE', data=raw_data, s=300, alpha = 0.3)

# WORK_GR @@가설검정: 조에 따라 스케일 발생률에 차이가 있는가?
raw_data['WORK_GR'] = np.where(raw_data['WORK_GR'] == '1조', str(1), np.where(raw_data['WORK_GR'] == '2조', str(2), np.where(raw_data['WORK_GR'] == '3조', str(3), str(4)))) # 조 번호만 남긴다.
raw_data['WORK_GR'].value_counts() # 균등하게 분배됨
sns.catplot(x='WORK_GR', y='SCALE', data = raw_data, kind='bar', height = 4, palette = 'muted') # 2조가 가장 스케일 발생이 낮음

############################## 2. 연속형 데이터를 살펴본다.##############################
''' 
[PT_THK', 'PT_WDTH', 'PT_LTH', 'PT_WGT', 'FUR_NO_ROW',
       'FUR_HZ_TEMP', 'FUR_HZ_TIME', 'FUR_SZ_TEMP', 'FUR_SZ_TIME', 'FUR_TIME',
       'FUR_EXTEMP', 'ROLLING_TEMP_T5', 'HSB', 'ROLLING_DESCALING', 'WORK_GR']
'''
data_cont = raw_data.select_dtypes(exclude='object')
# data_cont.drop(['HSB'], axis=1, inplace = True)
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

############################## 3. 가설 검정 ##############################
# 1) H0: FUR_NO에 따라 스케일 발생률에 차이가 없다.
raw_data['FUR_NO'].unique() # FUR_NO는 3개의 유니크한 값 확인
fur_no_result_f = f_oneway(raw_data[raw_data['FUR_NO']=='1']['SCALE'], \
                           raw_data[raw_data['FUR_NO']=='2']['SCALE'], \
                           raw_data[raw_data['FUR_NO']=='3']['SCALE'])

f, p = fur_no_result_f.statistic.round(3), fur_no_result_f.pvalue.round(3)
# 유의수준 5%에서 검정결과 P값이 0.201이므로 귀무가설을 채택. 집단간 차이가 없다.

# 2) H0: WORK_GR에 따라 스케일 발생률에 차이가 없다.
raw_data['WORK_GR'].unique() # FUR_NO는 4개의 유니크한 값 확인
work_gr_result_f = f_oneway(raw_data[raw_data['WORK_GR']=='1']['SCALE'], \
                           raw_data[raw_data['WORK_GR']=='2']['SCALE'], \
                           raw_data[raw_data['WORK_GR']=='3']['SCALE'], \
                           raw_data[raw_data['WORK_GR']=='4']['SCALE'])

f, p = work_gr_result_f.statistic.round(3), work_gr_result_f.pvalue.round(3)
# 유의수준 5%에서 검정결과 P값이 0.403이므로 귀무가설을 채택. 집단간 차이가 없다.

# 3) HO: FUR_NO_ROW에 따라 스케일 발생률에 차이가 없다.
raw_data['FUR_NO_ROW'].unique()
df1 = raw_data[raw_data['FUR_NO_ROW'] == 1]['SCALE']
df2 = raw_data[raw_data['FUR_NO_ROW'] == 2]['SCALE']
t_result = ttest_ind(df1, df2)

f, p = t_result.statistic.round(3), t_result.pvalue.round(3)
# 유의수준 5%에서 검정결과 P값이 0.521이므로 귀무가설을 채택. 집단간 차이가 없다.

# 4) H0: WORK_GR에 따라 스케일 발생률에 차이가 없다.
raw_data['WORK_GR'].unique() # FUR_NO는 4개의 유니크한 값 확인
work_gr_result_f = f_oneway(raw_data[raw_data['WORK_GR']=='1']['SCALE'], \
                           raw_data[raw_data['WORK_GR']=='2']['SCALE'], \
                           raw_data[raw_data['WORK_GR']=='3']['SCALE'], \
                           raw_data[raw_data['WORK_GR']=='4']['SCALE'])

f, p = work_gr_result_f.statistic.round(3), work_gr_result_f.pvalue.round(3)
# 유의수준 5%에서 검정결과 P값이 0.403이므로 귀무가설을 채택. 집단간 차이가 없다.