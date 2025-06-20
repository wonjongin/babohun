import pandas as pd


regions = ['중앙', '부산', '인천', '대구', '대전', '광주']
years = ['2019', '2020', '2021', '2022', '2023']
regions_df = {}

for region in regions:
    regions_df[region] = {}
    for year in years:
        regions_df[region][year] = pd.read_csv(f"data/비급여정보 UTF-8/보훈병원_비급여정보_{region}보훈병원_{year}.csv")
        regions_df[region][year]['년도'] = year
        if region != '중앙':
            regions_df[region][year]['지역'] = region
        else:
            regions_df[region][year]['지역'] = '서울'

rename_dict = {
    # 병원/기관
    '병원': None,

    # 구분/행위구분/상태 등
    '구분': '구분',
    '행위구분': '구분',
    '구분.1': None,
    '상태': None,
    '순번': None,
    '비고': None,

    # 분류
    '분류': '분류',
    '대분류': '대분류',
    '중분류': '중분류',
    '중분류명': '중분류',
    '소분류': '소분류',

    # 명칭
    '명칭': '명칭',
    '항목명칭': '명칭',
    '수가명칭': '명칭',
    '수가명': '명칭',
    # 'EDI명칭': '명칭',
    # '건강보험요양급여비용코드(EDI)명칭': '명칭',
    'EDI명칭': None,
    '건강보험요양급여비용코드(EDI)명칭': None,
    '의료기관사용명칭': None,
    '원내명칭': None,

    # 코드
    '코드': '코드',
    '항목코드': '코드',
    '수가코드': '코드',
    # 'EDI코드': '코드',
    # '건강보험요양급여비용코드(EDI)': '코드',
    'EDI코드': None,
    '건강보험요양급여비용코드(EDI)': None,
    '의료기관사용코드': None,
    '원내코드': None,
    '표준코드': None,

    # 비용
    '비용': '비용',
    ' 비용 ': '비용',
    '일반단가': '비용',

    # 최저/최고 비용
    '최저비용': '최저비용',
    ' 최저비용 ': '최저비용',
    '최고비용': '최고비용',
    ' 최고비용 ': '최고비용',

    # 치료재료대 포함여부
    '치료재료대 포함여부': '치료재료대포함여부',
    '치료재료대포함여부': '치료재료대포함여부',
    '치료재료대': '치료재료대포함여부',

    # 약제비 포함여부
    '약제비 포함여부': '약제비포함여부',
    '약제비포함여부': '약제비포함여부',

    # 특이사항
    '특이사항': '특이사항',

    # 날짜/등록일 등
    '년도': '년도',
    '지역': '지역',
    '최종변경일': None,
    '고지시작일': None,
    '고지종료일': None,
    '등록일': None,
}
drop_cols = [k for k, v in rename_dict.items() if v is None]


for region in regions:
    for year in years:
        # print(f"{region}:{year}:{regions_df[region][year].columns}")
        regions_df[region][year].replace('NA', pd.NA, inplace=True)
        regions_df[region][year].rename(columns={k: v for k, v in rename_dict.items() if v is not None}, inplace=True)
        regions_df[region][year] = regions_df[region][year].drop(columns=drop_cols, errors='ignore')



all_dfs = []

for region in regions:
    for year in years:
        regions_df[region][year] = regions_df[region][year].reset_index(drop=True)
        print(f"===== {region}:{year} =====")
        print(regions_df[region][year])
        print(regions_df[region][year].columns.duplicated())
        all_dfs.append(regions_df[region][year])

merged_df = pd.concat(all_dfs, ignore_index=True)

df = merged_df.replace('NA', pd.NA)

# df['구분'] = df['구분'].replace({
#     '입원 실인원': '입원(실인원)',
#     '입원 연인원': '입원(연인원)',
# })




max_min = "최저비용,최고비용".split(',')
df[max_min] = df[max_min].apply(pd.to_numeric, errors='coerce').astype('Int64')

# nan_mask = df["실인원"].isna()
# contains_mask = df['구분'].str.contains("실인원", na=False) | df['구분'].str.contains("외래", na=False)
# combined_mask = nan_mask & contains_mask
# df.loc[combined_mask, "실인원"] = df.loc[combined_mask, age_columns].sum(axis=1)

# ## 원래 위에만인데 인천 문제로 인천만 NA없어도 다 합계 다시계산
# nan_mask = (df["지역"] == "인천") & (df["년도"] == 2022)
# contains_mask = df['구분'].str.contains("외래", na=False)
# combined_mask = contains_mask
# df.loc[combined_mask, "실인원"] = df.loc[combined_mask, age_columns].sum(axis=1)

# nan_mask = df["연인원"].isna()
# contains_mask = df['구분'].str.contains("연인원", na=False)
# combined_mask = nan_mask & contains_mask
# df.loc[combined_mask, "연인원"] = df.loc[combined_mask, age_columns].sum(axis=1)



# df["연인원"] = pd.to_numeric(df["연인원"], errors='coerce').astype('Int64')
# df["실인원"] = pd.to_numeric(df["실인원"], errors='coerce').astype('Int64')
# df['연령별_합계_계산'] = df[age_columns].sum(axis=1).astype('Int64')
# print(df["실인원"].dtype)
# print(df["연인원"].dtype)
# print(df['연령별_합계_계산'].dtype)


# df['합_일치'] = (df["실인원"] == df['연령별_합계_계산']) | (df["연인원"] == df['연령별_합계_계산'])
# 전체_일치 = df['합_일치'].all()
# print(전체_일치)
# # for col in df.columns:
# #     print(f"========= Column: {col} =========")
# #     print(df[col].unique())


# 비용이 비어있고 최대비용 최소비용는 비어있지 않은 경우 -> 비용을 최대비용, 최소비용의 평균으로 대체

df['비용'] = df.apply(
    lambda row: (row['최저비용'] + row['최고비용']) / 2 if pd.isna(row['비용']) and not pd.isna(row['최저비용']) and not pd.isna(row['최고비용']) else row['비용'],
    axis=1
)
# 비용이 비어있고 최대비용 최소비용도 비어있는 경우 -> 비용을 0으로 대체
df['비용'] = df.apply(
    lambda row: 0 if pd.isna(row['비용']) and pd.isna(row['최저비용']) and pd.isna(row['최고비용']) else row['비용'],
    axis=1
)

# 비용이 비어있는데 최대랑 최소중에 하나만 비어있는 경우 -> 비용을 비어있지 않은 값으로 대체
df['비용'] = df.apply(
    lambda row: row['최저비용'] if pd.isna(row['비용']) and not pd.isna(row['최저비용']) and pd.isna(row['최고비용']) else row['비용'],
    axis=1
)
df['비용'] = df.apply(
    lambda row: row['최고비용'] if pd.isna(row['비용']) and pd.isna(row['최저비용']) and not pd.isna(row['최고비용']) else row['비용'],
    axis=1
)



cols = "년도,지역,분류,명칭,코드,구분,비용,최저비용,최고비용,치료재료대포함여부,약제비포함여부,특이사항,대분류,중분류,소분류".split(',')
df = df[cols]
df.to_csv('result_utf8.csv', index=False, encoding='utf-8')

# merged_df.to_csv('result_utf8.csv', index=False, encoding='utf-8')
