import pandas as pd
import numpy as np


df = pd.read_csv('merged_data/월별 진료과별 퇴원_열병합.csv')

df['월'] = df['월'].str.split('-').str[1]


columns = "월,퇴원과,퇴원결과,건수,지역,년도,주상병코드,주상병명".split(',')
df = df[columns]

for col in df.columns:
    print(f"========= Column: {col} =========")
    print(df[col].unique())

df.to_csv('월별 진료과별 퇴원_열병합_월구분.csv', index=False, encoding='utf-8')