import pandas as pd
import numpy as np

# 데이터 파일 불러오기 #
def safe_read_csv(path):
    try:
        return pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='cp949')

def add_year_region(df, year, region) :
    df['년도'] = year
    df['지역'] = region
    return df

def process_file(region, year, filepath):
    df = safe_read_csv(filepath)
    df = add_year_region(df, year, region)
    return df
    
bedInfo_Gwangju_2019 = process_file('광주', 2019, 'data/병상정보/보훈병원_병상정보_광주보훈병원_2019.csv')
bedInfo_Gwangju_2020 = process_file('광주', 2020, 'data/병상정보/보훈병원_병상정보_광주보훈병원_2020.csv')
bedInfo_Gwangju_2021 = process_file('광주', 2021, 'data/병상정보/보훈병원_병상정보_광주보훈병원_2021.csv')
bedInfo_Gwangju_2022 = process_file('광주', 2022, 'data/병상정보/보훈병원_병상정보_광주보훈병원_2022.csv')
bedInfo_Gwangju_2023 = process_file('광주', 2023, 'data/병상정보/보훈병원_병상정보_광주보훈병원_2023.csv')

bedInfo_Daegu_2019 = process_file('대구', 2019, 'data/병상정보/보훈병원_병상정보_대구보훈병원_2019.csv')
bedInfo_Daegu_2020 = process_file('대구', 2020, 'data/병상정보/보훈병원_병상정보_대구보훈병원_2020.csv')
bedInfo_Daegu_2021 = process_file('대구', 2021, 'data/병상정보/보훈병원_병상정보_대구보훈병원_2021.csv')
bedInfo_Daegu_2022 = process_file('대구', 2022, 'data/병상정보/보훈병원_병상정보_대구보훈병원_2022.csv')
bedInfo_Daegu_2023 = process_file('대구', 2023, 'data/병상정보/보훈병원_병상정보_대구보훈병원_2023.csv')

bedInfo_Daejeon_2019 = process_file('대전', 2019, 'data/병상정보/보훈병원_병상정보_대전보훈병원_2019.csv')
bedInfo_Daejeon_2020 = process_file('대전', 2020, 'data/병상정보/보훈병원_병상정보_대전보훈병원_2020.csv')
bedInfo_Daejeon_2021 = process_file('대전', 2021, 'data/병상정보/보훈병원_병상정보_대전보훈병원_2021.csv')
bedInfo_Daejeon_2022 = process_file('대전', 2022, 'data/병상정보/보훈병원_병상정보_대전보훈병원_2022.csv')
bedInfo_Daejeon_2023 = process_file('대전', 2023, 'data/병상정보/보훈병원_병상정보_대전보훈병원_2023.csv')

bedInfo_Busan_2019 = process_file('부산', 2019, 'data/병상정보/보훈병원_병상정보_부산보훈병원_2019.csv')
bedInfo_Busan_2020 = process_file('부산', 2020, 'data/병상정보/보훈병원_병상정보_부산보훈병원_2020.csv')
bedInfo_Busan_2021 = process_file('부산', 2021, 'data/병상정보/보훈병원_병상정보_부산보훈병원_2021.csv')
bedInfo_Busan_2022 = process_file('부산', 2022, 'data/병상정보/보훈병원_병상정보_부산보훈병원_2022.csv')
bedInfo_Busan_2023 = process_file('부산', 2023, 'data/병상정보/보훈병원_병상정보_부산보훈병원_2023.csv')

bedInfo_Incheon_2019 = process_file('인천', 2019, 'data/병상정보/보훈병원_병상정보_인천보훈병원_2019.csv')
bedInfo_Incheon_2020 = process_file('인천', 2020, 'data/병상정보/보훈병원_병상정보_인천보훈병원_2020.csv')
bedInfo_Incheon_2021 = process_file('인천', 2021, 'data/병상정보/보훈병원_병상정보_인천보훈병원_2021.csv')
bedInfo_Incheon_2022 = process_file('인천', 2022, 'data/병상정보/보훈병원_병상정보_인천보훈병원_2022.csv')
bedInfo_Incheon_2023 = process_file('인천', 2023, 'data/병상정보/보훈병원_병상정보_인천보훈병원_2023.csv')

bedInfo_Seoul_2019 = process_file('서울', 2019, 'data/병상정보/보훈병원_병상정보_중앙보훈병원_2019.csv')
bedInfo_Seoul_2020 = process_file('서울', 2020, 'data/병상정보/보훈병원_병상정보_중앙보훈병원_2020.csv')
bedInfo_Seoul_2021 = process_file('서울', 2021, 'data/병상정보/보훈병원_병상정보_중앙보훈병원_2021.csv')
bedInfo_Seoul_2022 = process_file('서울', 2022, 'data/병상정보/보훈병원_병상정보_중앙보훈병원_2022.csv')
bedInfo_Seoul_2023 = process_file('서울', 2023, 'data/병상정보/보훈병원_병상정보_중앙보훈병원_2023.csv')

# 열 이름 통일 #
dfs = [
    bedInfo_Gwangju_2019, bedInfo_Gwangju_2020, bedInfo_Gwangju_2021, bedInfo_Gwangju_2022, bedInfo_Gwangju_2023,
    bedInfo_Daegu_2019, bedInfo_Daegu_2020, bedInfo_Daegu_2021, bedInfo_Daegu_2022, bedInfo_Daegu_2023,
    bedInfo_Daejeon_2019, bedInfo_Daejeon_2020, bedInfo_Daejeon_2021, bedInfo_Daejeon_2022, bedInfo_Daejeon_2023,
    bedInfo_Busan_2019, bedInfo_Busan_2020, bedInfo_Busan_2021, bedInfo_Busan_2022, bedInfo_Busan_2023,
    bedInfo_Incheon_2019, bedInfo_Incheon_2020, bedInfo_Incheon_2021, bedInfo_Incheon_2022, bedInfo_Incheon_2023,
    bedInfo_Seoul_2019, bedInfo_Seoul_2020, bedInfo_Seoul_2021, bedInfo_Seoul_2022, bedInfo_Seoul_2023
]

for df in [bedInfo_Seoul_2021, bedInfo_Seoul_2022, bedInfo_Seoul_2023]:
    if '구분' in df.columns:
        df.rename(columns={'구분': '병동구분'}, inplace=True)

def normalize_column_name(col_name):
    if isinstance(col_name, str):
        col_name = col_name.strip()
        col_name = col_name.replace('\xa0', '')
        col_name = col_name.replace('\u3000', '')
        col_name = col_name.replace('\u200b', '')
        col_name = col_name.replace(' ', '')
    return col_name

columns_map = {
    '구분': '병상 구분', '병상현황': '병상 구분', '병상구분': '병상 구분',
    '구      분 ': '병상 구분', ' 구      분 ': '병상 구분', ' 구분 ': '병상 구분',
    '병동구분': '병동 구분', '병실형태': '병동 구분', ' 병동구분 ': '병동 구분',
    '병상합계': '병상 합계', '합계병상': '병상 합계',
    '병실합계': '병실 합계', '합계병실': '병실 합계',
    '1인실병상': '1인실 병상', ' 1인실 병상 ': '1인실 병상',
    '1인실병실': '1인실 병실', ' 1인실 병실 ': '1인실 병실',
    '2인실병상': '2인실 병상', ' 2인실 병상 ': '2인실 병상',
    '2인실병실': '2인실 병실', ' 2인실 병실 ': '2인실 병실',
    '3인실병상': '3인실 병상', ' 3인실 병상 ': '3인실 병상',
    '3인실병실': '3인실 병실', ' 3인실 병실 ': '3인실 병실',
    '4인실병상': '4인실 병상', ' 4인실 병상 ': '4인실 병상', '4인 실 병상': '4인실 병상',
    '4인실병실': '4인실 병실', ' 4인실 병실 ': '4인실 병실', '4 인실병실': '4인실 병실',
    '5인실병상': '5인실 병상', ' 5인실 병상 ': '5인실 병상',
    '5인실병실': '5인실 병실', ' 5인실 병실 ': '5인실 병실',
    '6인실병상': '6인실 병상', ' 6인실 병상 ': '6인실 병상', '6인 실 병상': '6인실 병상',
    '6인실병실': '6인실 병실', '6 인실 병실': '6인실 병실', ' 6인실 병실 ': '6인실 병실',
    '7인실 병상': '7인실 병상', ' 7인실 병상 ': '7인실 병상', ' 7인실 병실 ': '7인실 병실',
    '8인실 병상': '8인실 병상', ' 8인실 병상 ': '8인실 병상', ' 8인실 병실 ': '8인실 병실',
    '9인실 병상': '9인실 병상', ' 9인실 병상 ': '9인실 병상', ' 9인실 병실 ': '9인실 병실',
    '특실병상': '특실 병상', ' 특실 병상 ': '특실 병상',
    '특실병실': '특실 병실', ' 특실 병실 ': '특실 병실',
    '기타병상': '기타 병상', ' 기타 병상 ': '기타 병상',
    '기타병실': '기타 병실', ' 기타 병실 ': '기타 병실',
    '격리실 병상': '격리실 병상', ' 격리실 병상 ': '격리실 병상', ' 격리실 병실 ': '격리실 병실',
    '국비 병상': '국비 병상', ' 국비 병상 ': '국비 병상', ' 국비 병실 ': '국비 병실',
    '사비 병상': '사비 병상', ' 사비 병상 ': '사비 병상', ' 사비 병실 ': '사비 병실'
}

normalized_columns_map = {}
for k, v in columns_map.items():
    nk = normalize_column_name(k)
    normalized_columns_map[nk] = v

def rename_columns(df, col_map):
    new_columns = []
    for col in df.columns:
        col_norm = normalize_column_name(col)
        new_col = col_map.get(col_norm, col)
        new_columns.append(new_col)
    df.columns = new_columns
    return df

def strip_string_values(df):
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip()
    return df

for i, df in enumerate(dfs):
    dfs[i] = rename_columns(df, normalized_columns_map)
    dfs[i] = strip_string_values(dfs[i])
    print(f"[{i}] 열 이름: {dfs[i].columns.tolist()}")

target_columns = [
    '1인실 병상', '1인실 병실', '2인실 병상', '2인실 병실','3인실 병상', '3인실 병실', '4인실 병상', '4인실 병실',
    '5인실 병상', '5인실 병실', '6인실 병상', '6인실 병실', '7인실 병상', '7인실 병실',
    '8인실 병상', '8인실 병실', '9인실 병상', '9인실 병실', '기타 병상', '기타 병실', '특실 병상', '특실 병실',
    '격리실 병상', '격리실 병실', '국비 병상', '국비 병실', '사비 병상', '사비 병실'
]

exclude_from_fillna = ['국비 병상', '국비 병실', '사비 병상', '사비 병실']

for df in dfs:
    for col in target_columns:
        if col not in df.columns:
            df[col] = np.nan

        if col not in exclude_from_fillna:
            df[col] = df[col].fillna(0).astype(int)


bed_columns = [col for col in target_columns if '병상' in col and not col.startswith(('국비', '사비'))]
room_columns = [col for col in target_columns if '병실' in col and not col.startswith(('국비', '사비'))]

for df in dfs:
    df['병상 합계'] = df[bed_columns].sum(axis=1)
    df['병실 합계'] = df[room_columns].sum(axis=1)

merged_df = pd.concat(dfs, ignore_index=True)

values = merged_df['병동 구분'].dropna().unique()
values_sorted = sorted(values)

for v in values_sorted:
    print(v)

merged_df['병동 구분'] = merged_df['병동 구분'].astype(str).str.strip().str.lower()

mapping = {
     '51병동': '일반병동',
    '61병동': '일반병동',
    '71병동': '일반병동',
    '일반병실': '일반병동',

    '간호간병': '간호간병병동',

    '격리병실': '격리병실',
    '음압격리실': '음압격리실',
    '음압실': '음압격리리실',

    '무균치료실': '무균치료실',
    '물리치료실': '물리치료실',
    '분만실': '분만실',
    '수술실': '수술실',
    '신생아실': '신생아실',
    '완화의료': '완화의료실',
    '응급실': '응급실',
    '인공신장실': '인공신장실',
    '재활병동': '재활병동',
    '정신과폐쇄': '정신과폐쇄병동',
    '중환자': '중환자실',
    '중환자실': '중환자실',
    '회복실': '회복실',

    # 코로나 관련
    '코로나 전담병동': '코로나전담병동',
    '코로나19전담병동': '코로나전담병동',
    '코로나전담병동': '코로나전담병동',

    # 합계/계
    '계': None
}

merged_df = merged_df[merged_df['병동 구분'] != '계']
merged_df['병동 구분'] = merged_df['병동 구분'].map(mapping).fillna(merged_df['병동 구분'])

print(merged_df['병동 구분'].unique())

merged_df.to_csv('C:/Users/julia/Downloads/병상정보.csv', index=False, encoding='UTF-8')