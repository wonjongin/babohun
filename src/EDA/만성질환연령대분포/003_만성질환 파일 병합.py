import pandas as pd
import numpy as np


df = pd.read_csv('merged_data/만성질환 환자 연령별 현황.csv')
# print(df.shape)                         # 데이터 크기(행, 열)
# print(df.columns)                       # 컬럼명
# print(df.dtypes)                        # 데이터 타입
# print(df.head())                        # 상위 5개 행 미리보기
# print(df.info())                        # 전체 요약 정보(결측치, 타입 등)
# print(df.describe())                    # 수치형 변수의 기초 통계량
# print(df.describe(include='object'))    # 범주형 변수의 기초 통계량
print(df['구분'].unique())
df['구분'] = df['구분'].replace({
    '입원 실인원': '입원(실인원)',
    '입원 연인원': '입원(연인원)'
})

df = df.replace('NA', np.nan)


# df['merged'] = df['col1'].fillna(df['col2'])

related_columns = [
    ["X59이하", "연령별.59이하."], 
    ["X60.64", "연령별.60.64."],
    ["X65.69", "연령별.65.69."], 
    ["X70.79", "연령별.70.79."], 
    ["X80.89", "연령별.80.89."], 
    ["X90이상", "연령별.90이상."],
    ["코드", "상병코드"],
    ["상병명", "상병명.명칭"],]

for col1, col2 in related_columns:
    df[col1] = df[col1].fillna(df[col2])
    df.drop(columns=[col2], inplace=True)


df['상병명'] = df['상병명'].fillna(df['명칭'])
df.drop(columns=['명칭'], inplace=True)

age_columns = ['X59이하', 'X60.64', 'X65.69', 'X70.79', 'X80.89', 'X90이상']
df[age_columns] = df[age_columns].apply(pd.to_numeric, errors='coerce').astype('Int64')
df.loc[df['연령별.합계'].isna(), '연령별.합계'] = df.loc[df['연령별.합계'].isna(), age_columns].sum(axis=1)

df = df[['년도', '지역', '구분', '코드', '상병명', '연령별.합계', 'X59이하', 'X60.64', 'X65.69', 'X70.79', 'X80.89', 'X90이상']]

print(df.columns)
df.replace(np.nan, 'NaN', inplace=True)
df.to_csv('result_utf8.csv', index=False, encoding='utf-8')


'''
 0   구분          561 non-null    object 
 1   코드          429 non-null    object 
 2   상병명         429 non-null    object 
 3   연령별.합계      198 non-null    object 
 4   X59이하       308 non-null    object 
 5   X60.64      296 non-null    object 
 6   X65.69      300 non-null    object 
 7   X70.79      322 non-null    object 
 8   X80.89      318 non-null    object 
 9   X90이상       300 non-null    object 
 10  지역          561 non-null    object 
 11  년도          561 non-null    int64  
 12  연령별.59이하.   155 non-null    float64
 13  연령별.60.64.  146 non-null    float64
 14  연령별.65.69.  148 non-null    float64
 15  연령별.70.79.  165 non-null    float64
 16  연령별.80.89.  157 non-null    float64
 17  연령별.90이상.   154 non-null    float64
 18  명칭          66 non-null     object 
 19  상병코드        66 non-null     object 
 20  상병명.명칭      66 non-null     object 
'''