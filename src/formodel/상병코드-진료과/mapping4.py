import pandas as pd

map_df = pd.read_csv("new_merged_data/상병코드_진료과_매핑3.csv", dtype=str)

map_df.columns = map_df.columns.str.strip()

# 딕셔너리 생성
dept_map = dict(zip(map_df['상병코드'], map_df['진료과']))

df = pd.read_csv("new_merged_data/df_result2.csv", dtype=str)
df['상병코드'] = df['상병코드'].str.strip().str[:3]

# 4) 우선 map_df로 매핑, 없으면 icd_ranges로 매핑, 그래도 없으면 '미분류'
def get_dept(code):
    if code in dept_map:
        return dept_map[code]

df['진료과'] = df['상병코드'].apply(get_dept)

# 5) 결과 확인 및 저장
print(df[['상병코드','진료과']].drop_duplicates().head(20))
df.to_csv("new_merged_data/df_result2_with_심평원.csv", index=False)