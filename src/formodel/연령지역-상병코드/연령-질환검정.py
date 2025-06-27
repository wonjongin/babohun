import pandas as pd
from scipy.stats import chi2_contingency

## 연령 질환 독립성 검정
# 1) 데이터 불러오기 및 재구조화
df = pd.read_csv("new_merged_data/df_result2.csv")

age_cols = ['59이하','60-64','65-69','70-79','80-89','90이상']

# 2) melt & NA 처리
melted = df.melt(
    id_vars=['년도','상병코드'],
    value_vars=age_cols,
    var_name='age_group',
    value_name='count'
)
melted['count'] = melted['count'].fillna(0)

# 3) 전체 검정: 교차표에 sum 집계 지정
ct_all = pd.crosstab(
    index=melted['age_group'],
    columns=melted['상병코드'],
    values=melted['count'],
    aggfunc='sum'
)
chi2_all, p_all, dof_all, _ = chi2_contingency(ct_all)
print(f"전체: Chi2={chi2_all:.4f}, p-value={p_all:.4f}, df={dof_all}")

# 4) 연도별 검정
for yr in sorted(melted['년도'].unique()):
    sub = melted[melted['년도']==yr]
    ct = pd.crosstab(
        index=sub['age_group'],
        columns=sub['상병코드'],
        values=sub['count'],
        aggfunc='sum'
    )
    chi2, p, dof, _ = chi2_contingency(ct)
    print(f"{yr}년: Chi2={chi2:.4f}, p-value={p:.4f}, df={dof}")

