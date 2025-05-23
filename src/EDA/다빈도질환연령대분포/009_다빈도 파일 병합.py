import pandas as pd


regions = ['중앙', '부산', '인천', '대구', '대전', '광주']
years = ['2021', '2022', '2023']
regions_df = {}

for region in regions:
    regions_df[region] = {}
    for year in years:
        regions_df[region][year] = pd.read_csv(f"modified_data/다빈도 질환 환자 연령대별 현황/다빈도 질환 환자 연령별 분포_{region}보훈병원_{year}.csv")
        regions_df[region][year]['년도'] = year
        if region != '중앙':
            regions_df[region][year]['지역'] = region
        else:
            regions_df[region][year]['지역'] = '서울'

rename_dict = {
    # 공통
    '상병명칭': '상병명',
    # '상병명': '상병명',
    '명칭': '상병명',
    # '상병코드': '상병코드',
    '코드': '상병코드',
    '상병명 코드': '상병코드',
    # '실인원': '실인원',
    '진료인원(실인원)': '실인원',
    # '연인원': '연인원',
    '진료인원(연인원)': '연인원',
    # '진료비(천원)': None,  # 필요 없으니 무시
    # '년도': '년도',
    # '지역': '지역',
    # '구분': '구분',
    # '순위': '순위',

    # 연령대
    '59이하': '59이하',
    '59세이하': '59이하',
    '연령별 59이하': '59이하',

    '60-64': '60-64',
    '60세-64세': '60-64',
    '연령별(60-64)': '60-64',

    '65-69': '65-69',
    '65세-69세': '65-69',
    '연령별(65-69)': '65-69',

    '70-79': '70-79',
    '70세-79세': '70-79',
    '연령별(70-79)': '70-79',

    '80-89': '80-89',
    '80세-89세': '80-89',
    '연령별(80-89)': '80-89',

    '90이상': '90이상',
    '90세이상': '90이상',
    '연령별(90이상)': '90이상',
}



for region in regions:
    for year in years:
        print(f"{region}:{year}:{regions_df[region][year].columns}")
        regions_df[region][year].replace('NA', pd.NA, inplace=True)
        regions_df[region][year].rename(columns=rename_dict, inplace=True)


all_dfs = []

for region in regions:
    for year in years:
        all_dfs.append(regions_df[region][year])

merged_df = pd.concat(all_dfs, ignore_index=True)

df = merged_df.replace('NA', pd.NA)

df['구분'] = df['구분'].replace({
    '입원 실인원': '입원(실인원)',
    '입원 연인원': '입원(연인원)',
})

age_columns = "59이하,60-64,65-69,70-79,80-89,90이상".split(',')
df[age_columns] = df[age_columns].apply(pd.to_numeric, errors='coerce').astype('Int64')

nan_mask = df["실인원"].isna()
contains_mask = df['구분'].str.contains("실인원", na=False) | df['구분'].str.contains("외래", na=False)
combined_mask = nan_mask & contains_mask
df.loc[combined_mask, "실인원"] = df.loc[combined_mask, age_columns].sum(axis=1)

## 원래 위에만인데 인천 문제로 인천만 NA없어도 다 합계 다시계산
nan_mask = (df["지역"] == "인천") & (df["년도"] == 2022)
contains_mask = df['구분'].str.contains("외래", na=False)
combined_mask = contains_mask
df.loc[combined_mask, "실인원"] = df.loc[combined_mask, age_columns].sum(axis=1)

nan_mask = df["연인원"].isna()
contains_mask = df['구분'].str.contains("연인원", na=False)
combined_mask = nan_mask & contains_mask
df.loc[combined_mask, "연인원"] = df.loc[combined_mask, age_columns].sum(axis=1)



df["연인원"] = pd.to_numeric(df["연인원"], errors='coerce').astype('Int64')
df["실인원"] = pd.to_numeric(df["실인원"], errors='coerce').astype('Int64')
df['연령별_합계_계산'] = df[age_columns].sum(axis=1).astype('Int64')
print(df["실인원"].dtype)
print(df["연인원"].dtype)
print(df['연령별_합계_계산'].dtype)


df['합_일치'] = (df["실인원"] == df['연령별_합계_계산']) | (df["연인원"] == df['연령별_합계_계산'])
전체_일치 = df['합_일치'].all()
print(전체_일치)
# for col in df.columns:
#     print(f"========= Column: {col} =========")
#     print(df[col].unique())

print(df.columns)
df = df[['년도', '지역', '구분', '순위', '상병코드', '상병명', '실인원', '연인원', '59이하', '60-64', '65-69', '70-79', '80-89', '90이상', '진료비(천원)']]

df.to_csv('result_utf8.csv', index=False, encoding='utf-8')

# merged_df.to_csv('result_utf8.csv', index=False, encoding='utf-8')
