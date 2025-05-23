import pandas as pd
import numpy as np


df = pd.read_csv('merged_data/만성질환 환자 연령별 현황_인천포함.csv')

df = df.replace('NA', pd.NA)

df.replace(pd.NA, 0, inplace=True)
df.to_csv('result_utf8.csv', index=False, encoding='utf-8')
