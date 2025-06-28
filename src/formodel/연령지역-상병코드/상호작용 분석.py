import pandas as pd
df_result2_mapping1=pd.read_csv("C:/Users/jenny/OneDrive - dgu.ac.kr/바탕 화면/df_result2_mapping1.csv",encoding="utf-8-sig")

#지역 연령대 상호작용
age_cols = ['59이하', '60-64', '65-69', '70-79', '80-89', '90이상']
# 1) long-form 데이터 준비
df_long = df_result2_mapping1.melt(
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
