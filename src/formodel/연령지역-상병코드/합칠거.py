# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 15:35:17 2025

@author: jenny
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import statsmodels.api as sm


ekqlseh=pd.read_csv("C:/Users/jenny/babohun/new_merged_data/다빈도 질환 환자 연령별 분포_순위추가_합계계산_값통일.csv",encoding="utf-8-sig")
ekqlseh.loc[ekqlseh['구분'].str.contains('외래'), '연인원'] = \
    ekqlseh.loc[ekqlseh['구분'].str.contains('외래'), '실인원']
mask = ekqlseh['구분'] == '입원(실인원)'
rows_to_drop = ekqlseh[ekqlseh['구분'] == '입원(실인원)'].index
df_dropped = ekqlseh.drop(index=rows_to_drop)
cols_to_drop = ['순위', '상병명', '실인원', '진료비(천원)']
df_result = df_dropped.drop(columns=cols_to_drop)
df_result
df_result.to_csv(
    "C:/Users/jenny/babohun/new_merged_data/df_result2.csv",
    index=False,
    encoding="utf-8-sig")

### 지역-질병 독립성 검정

ct = pd.crosstab(df_result['지역'],df_result['상병코드'],values=df_result['연인원'], aggfunc='sum').fillna(0)
ct = ct.loc[ct.sum(axis=1) > 0, ct.sum(axis=0) > 0]
prob = ct.div(ct.sum(axis=1), axis=0)
print(prob)
chi2, p, dof, expected = chi2_contingency(ct)

print(f"Chi² 통계량: {chi2:.3f}")
print(f"p-value: {p:.3e}")
print(f"자유도(df): {dof}")

chi2_stat, p, dof, expected = chi2_contingency(ct)
n = ct.values.sum()
min_dim = min(ct.shape) - 1
cramer_v = np.sqrt(chi2_stat / (n * min_dim))
print("Cramer's V:", cramer_v)

'''
Chi² 통계량: 8189981.212
p-value: 0.000e+00
자유도(df): 990
Cramer's V: 0.5299659786175956
귀무가설(특정 질병은 특정 지역에서 많이 걸리지 않는다) 기각
'''
#나눠서

for year, grp in df_result.groupby('년도'):
    print(f"\n=== {year}년 분석 ===")
    ct = pd.crosstab(grp['지역'], grp['상병코드'], values=grp['연인원'], aggfunc='sum').fillna(0)
    ct = ct.loc[ct.sum(axis=1) > 0, ct.sum(axis=0) > 0]
    prob = ct.div(ct.sum(axis=1), axis=0)
    print("지역별 질병 발생확률 (첫 5개 지역):")
    print(prob.head(), "\n")
    chi2_stat, p_value, dof, expected = chi2_contingency(ct)
    print(f"Chi² 통계량: {chi2_stat:.3f}")
    print(f"p-value: {p_value:.3e}")
    print(f"자유도(df): {dof}")
    n = ct.values.sum()
    min_dim = min(ct.shape) - 1
    cramer_v = np.sqrt(chi2_stat / (n * min_dim))
    print(f"Cramer's V: {cramer_v:.4f}")


#지역 연령대 상호작용
age_cols = ['50-54', '55-59', '60-64', '65-69', '70-79', '80-89', '90이상']

# 1) long-form 데이터 준비
df_long = df_result.melt(
    id_vars=['지역'], 
    value_vars=age_cols, 
    var_name='age_group', 
    value_name='count'
)

# 2) 디자인 매트릭스: interaction 포함
y, X = pt.dmatrices(
    'count ~ C(지역) + C(age_group) + C(지역):C(age_group)',
    data=df_long,
    return_type='dataframe'
)

# 3) Poisson GLM 적합
model = sm.GLM(y, X, family=sm.families.Poisson())
result = model.fit()

print(result.summary())

print([repr(c) for c in df_result.columns.tolist()])
