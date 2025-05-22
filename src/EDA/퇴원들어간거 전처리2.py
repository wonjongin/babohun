import pandas as pd
import numpy as np

def validate_na_value_pairs(df, col1, col2):
    condition = ((df[col1].isna() & df[col2].notna()) | (df[col1].notna() & df[col2].isna()))
    if not condition.all():
        return False, f"{col1}와 {col2} 열의 조건을 만족하지 않는 행이 있습니다."
    return True, "모든 행이 두 열 중 하나는 NA이고 다른 하나는 값이 들어있습니다."

df = pd.read_csv('merged_data/과별 퇴원환자 20대 주진단.csv')

# print(df['구분'].unique())
# df['구분'] = df['구분'].replace({
#     '입원 실인원': '입원(실인원)',
#     '입원 연인원': '입원(연인원)'
# })

df = df.replace('NA', pd.NA)


related_columns = [
    ["진료과", "부서명"],
    ["상병명", "주진단명"]]

for col1, col2 in related_columns:
    if not validate_na_value_pairs(df, col1, col2):
        exit(1)


for col1, col2 in related_columns:
    df[col1] = df[col1].fillna(df[col2])
    df.drop(columns=[col2], inplace=True)

# print(df.columns)
df.replace(np.nan, 'NA', inplace=True)
df.to_csv('과별 퇴원환자 20대 주진단_열병합.csv', index=False, encoding='utf-8')