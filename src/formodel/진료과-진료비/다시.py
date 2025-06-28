import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
import scipy.stats as stats
from scipy.stats import brunnermunzel
from scipy.stats import skew, kurtosis
from scipy.stats import normaltest

# 원본 데이터 불러오기
ekqlseh = pd.read_csv("C:/Users/jenny/babohun/new_merged_data/다빈도 질환 환자 연령별 분포_순위추가_합계계산_값통일.csv",encoding="utf-8-sig")
# 외래인원 컬럼 병합, 입원(실인원) 행 제거
ekqlseh.loc[ekqlseh['구분'].str.contains('외래'), '연인원'] = ekqlseh['실인원']
ekqlseh = ekqlseh[~(ekqlseh['구분'] == '입원(실인원)')]
# 불필요한 컬럼 제거
df_result = ekqlseh.drop(columns=['순위', '상병명', '실인원'])
# 지역 필터링
exclude_regions = ['서울', '대전', '대구']
df_filtered = df_result[~df_result['지역'].isin(exclude_regions)].copy()
# 상병코드 ↔ 진료과 매핑 테이블 불러오기 및 병합
mapping = pd.read_csv("C:/Users/jenny/babohun/df_result2_with_심평원.csv",encoding="utf-8-sig")
df_filtered = (df_filtered.merge(mapping[['상병코드', '진료과']], on='상병코드', how='left').dropna(subset=['진료과'])) 
df_filtered.rename(columns={'진료비(천원)': '진료비'}, inplace=True)
###################################################################################################################################################################
#정규성 검정
print("=== 정규성 검정 (Shapiro–Wilk) by 진료과 ===")
for dept, grp in df_filtered.groupby('진료과'):
    vals = grp['진료비'].dropna().values
    n = len(vals)
    if n < 3:
        print(f"{dept}: n={n} (<3), 검정 생략")
        continue
    if n > 500:
        vals = np.random.choice(vals, 500, replace=False)
    stat, p = stats.shapiro(vals)
    print(f"{dept}: W={stat:.4f}, p-value={p:.4e} (n={n})")

# 등분산성 검정 (Levene)
print("=== 등분산성 검정 (Levene) by 진료과 ===")
groups = [grp['진료비'].dropna().values
    for _, grp in df_filtered.groupby('진료과')
    if len(grp) >= 2]
stat, p = stats.levene(*groups, center='median')
print(f"Levene H={stat:.4f}, p-value={p:.4e}") 
###################################################################################################################################################################
# 더 비모수적인 등분산성 검정(Fligner–Killeen)
print("=== 등분산성 검정 (Fligner–Killeen) by 진료과 ===")
stat, p = stats.fligner(*groups)
print(f"Fligner–Killeen H={stat:.4f}, p-value={p:.4e}")

# Permutation ANOVA
print("=== 비모수 검정1 (Permutation ANOVA) by 진료과 ===")
def perm_anova(data, labels, n_perm=5000):
    obs_F = stats.f_oneway(
        *[data[labels == g] for g in np.unique(labels)]).statistic
    cnt = 0
    for _ in range(n_perm):
        perm = np.random.permutation(labels)
        F = stats.f_oneway(
            *[data[perm == g] for g in np.unique(labels)]).statistic
        if F >= obs_F:
            cnt += 1
    return obs_F, cnt / n_perm

costs = df_filtered['진료비'].values
labs  = df_filtered['진료과'].values
F, p = perm_anova(costs, labs, n_perm=10000)
print(f"Permutation ANOVA F={F:.4f}, p-value={p:.4f}")

# Brunner–Munzel 검정
print("=== 비모수 검정2 ( Brunner–Munzel) by 진료과 ===")
grp1 = df_filtered[df_filtered['진료과']=='내과']['진료비']
grp2 = df_filtered[df_filtered['진료과']=='안과']['진료비']
stat, p = brunnermunzel(grp1, grp2)
print(f"Brunner–Munzel W={stat:.4f}, p-value={p:.4e}")
###################################################################################################################################################################
# 왜곡도(skewness)·뾰족도(kurtosis) 계산
print("=== 왜곡도(skewness)·뾰족도(kurtosis) 계산 by 진료과 ===")
for dept, grp in df_filtered.groupby('진료과'):
    vals = grp['진료비'].dropna()
    sk = skew(vals)
    kt = kurtosis(vals)  # Fisher’s definition (normal → 0)
    print(f"{dept:<10}  skewness={sk:.2f},  kurtosis={kt:.2f}")

#  D’Agostino’s K² 검정 (normaltest)
print("=== D’Agostino’s K² 검정 (normaltest) by 진료과 ===")

for dept, grp in df_filtered.groupby('진료과'):
    vals = grp['진료비'].dropna()
    if len(vals) >= 8:  # normaltest 권장 최소치
        stat, p = normaltest(vals)
        print(f"{dept:<10}  K2={stat:.2f}, p-value={p:.4f}")

