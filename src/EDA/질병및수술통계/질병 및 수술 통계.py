import pandas as pd
import numpy as np

# 데이터 파일 불러오기 #
def safe_read_csv(path):
    try:
        return pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='cp949')
    
def add_year_region(df, year, region,) :
    df['년도'] = year
    df['지역'] = region
    return df

def process_file(region, year, filepath):
    df = safe_read_csv(filepath)
    df = add_year_region(df, year, region)

    # '코드'가 들어간 열 이름 찾기
    code_columns = [col for col in df.columns if '코드' in col]

    if code_columns:
        code_column = code_columns[0]  # 첫 번째 '코드' 열만 사용
        df['구분'] = df[code_column].astype(str).apply(
            lambda x: '질병' if any(c.isalpha() for c in x) else '수술'
        )
    else:
        print(f"⚠️ '코드'가 포함된 열이 '{filepath}'에 존재하지 않습니다.")

    return df

diseaseSurgeryStats_Gwangju_2019 = process_file('광주', 2019, 'data/질병 및 수술통계/보훈병원_질병수술통계_광주보훈병원_2019.csv')
diseaseSurgeryStats_Gwangju_2020 = process_file('광주', 2020, 'data/질병 및 수술통계/보훈병원_질병수술통계_광주보훈병원_2020.csv')
diseaseSurgeryStats_Gwangju_2021 = process_file('광주', 2021, 'data/질병 및 수술통계/보훈병원_질병수술통계_광주보훈병원_2021.csv')
diseaseSurgeryStats_Gwangju_2022 = process_file('광주', 2022, 'data/질병 및 수술통계/보훈병원_질병수술통계_광주보훈병원_2022.csv')
diseaseSurgeryStats_Gwangju_2023 = process_file('광주', 2023, 'data/질병 및 수술통계/보훈병원_질병수술통계_광주보훈병원_2023.csv')

diseaseSurgeryStats_Busan_2019 = process_file('부산', 2019, 'data/질병 및 수술통계/보훈병원_질병수술통계_부산보훈병원_2019.csv')
diseaseSurgeryStats_Busan_2020 = process_file('부산', 2020, 'data/질병 및 수술통계/보훈병원_질병수술통계_부산보훈병원_2020.csv')
diseaseSurgeryStats_Busan_2021 = process_file('부산', 2021, 'data/질병 및 수술통계/보훈병원_질병수술통계_부산보훈병원_2021.csv')
diseaseSurgeryStats_Busan_2022 = process_file('부산', 2022, 'data/질병 및 수술통계/보훈병원_질병수술통계_부산보훈병원_2022.csv')
diseaseSurgeryStats_Busan_2023 = process_file('부산', 2023, 'data/질병 및 수술통계/보훈병원_질병수술통계_부산보훈병원_2023.csv')

diseaseSurgeryStats_Daegu_2019 = process_file('대구', 2019, 'data/질병 및 수술통계/보훈병원_질병수술통계_대구보훈병원_2019.csv')
diseaseSurgeryStats_Daegu_2020 = process_file('대구', 2020, 'data/질병 및 수술통계/보훈병원_질병수술통계_대구보훈병원_2020.csv')
diseaseSurgeryStats_Daegu_2021 = process_file('대구', 2021, 'data/질병 및 수술통계/보훈병원_질병수술통계_대구보훈병원_2021.csv')
diseaseSurgeryStats_Daegu_2022 = process_file('대구', 2022, 'data/질병 및 수술통계/보훈병원_질병수술통계_대구보훈병원_2022.csv')
diseaseSurgeryStats_Daegu_2023 = process_file('대구', 2023, 'data/질병 및 수술통계/보훈병원_질병수술통계_대구보훈병원_2023.csv')

diseaseSurgeryStats_Daejeon_2019 = process_file('대전', 2019, 'data/질병 및 수술통계/보훈병원_질병수술통계_대전보훈병원_2019.csv')
diseaseSurgeryStats_Daejeon_2020 = process_file('대전', 2020, 'data/질병 및 수술통계/보훈병원_질병수술통계_대전보훈병원_2020.csv')
diseaseSurgeryStats_Daejeon_2021 = process_file('대전', 2021, 'data/질병 및 수술통계/보훈병원_질병수술통계_대전보훈병원_2021.csv')
diseaseSurgeryStats_Daejeon_2022 = process_file('대전', 2022, 'data/질병 및 수술통계/보훈병원_질병수술통계_대전보훈병원_2022.csv')
diseaseSurgeryStats_Daejeon_2023 = process_file('대전', 2023, 'data/질병 및 수술통계/보훈병원_질병수술통계_대전보훈병원_2023.csv')

diseaseSurgeryStats_Incheon_2019 = process_file('인천', 2019, 'data/질병 및 수술통계/보훈병원_질병수술통계_인천보훈병원_2019.csv')
diseaseSurgeryStats_Incheon_2020 = process_file('인천', 2020, 'data/질병 및 수술통계/보훈병원_질병수술통계_인천보훈병원_2020.csv')
diseaseSurgeryStats_Incheon_2021 = process_file('인천', 2021, 'data/질병 및 수술통계/보훈병원_질병수술통계_인천보훈병원_2021.csv')
diseaseSurgeryStats_Incheon_2022 = process_file('인천', 2022, 'data/질병 및 수술통계/보훈병원_질병수술통계_인천보훈병원_2022.csv')
diseaseSurgeryStats_Incheon_2023 = process_file('인천', 2023, 'data/질병 및 수술통계/보훈병원_질병수술통계_인천보훈병원_2023.csv')

diseaseSurgeryStats_Seoul_2019 = process_file('서울', 2019, 'data/질병 및 수술통계/보훈병원_질병수술통계_중앙보훈병원_2019.csv')
diseaseSurgeryStats_Seoul_2020 = process_file('서울', 2020, 'data/질병 및 수술통계/보훈병원_질병수술통계_중앙보훈병원_2020.csv')
diseaseSurgeryStats_Seoul_2021 = process_file('서울', 2021, 'data/질병 및 수술통계/보훈병원_질병수술통계_중앙보훈병원_2021.csv')
diseaseSurgeryStats_Seoul_2022 = process_file('서울', 2022, 'data/질병 및 수술통계/보훈병원_질병수술통계_중앙보훈병원_2022.csv')
diseaseSurgeryStats_Seoul_2023 = process_file('서울', 2023, 'data/질병 및 수술통계/보훈병원_질병수술통계_중앙보훈병원_2023.csv')

dfs = [
    diseaseSurgeryStats_Gwangju_2019, diseaseSurgeryStats_Gwangju_2020, diseaseSurgeryStats_Gwangju_2021, diseaseSurgeryStats_Gwangju_2022, diseaseSurgeryStats_Gwangju_2023,
    diseaseSurgeryStats_Busan_2019, diseaseSurgeryStats_Busan_2020, diseaseSurgeryStats_Busan_2021, diseaseSurgeryStats_Busan_2022, diseaseSurgeryStats_Busan_2023,
    diseaseSurgeryStats_Daegu_2019, diseaseSurgeryStats_Daegu_2020, diseaseSurgeryStats_Daegu_2021, diseaseSurgeryStats_Daegu_2022, diseaseSurgeryStats_Daegu_2023,
    diseaseSurgeryStats_Daejeon_2019, diseaseSurgeryStats_Daejeon_2020, diseaseSurgeryStats_Daejeon_2021, diseaseSurgeryStats_Daejeon_2022, diseaseSurgeryStats_Daejeon_2023,
    diseaseSurgeryStats_Incheon_2019, diseaseSurgeryStats_Incheon_2020, diseaseSurgeryStats_Incheon_2021, diseaseSurgeryStats_Incheon_2022, diseaseSurgeryStats_Incheon_2023,
    diseaseSurgeryStats_Seoul_2019, diseaseSurgeryStats_Seoul_2020, diseaseSurgeryStats_Seoul_2021, diseaseSurgeryStats_Seoul_2022, diseaseSurgeryStats_Seoul_2023
]

def normalize_column_name(name):
    invisible_chars = ''.join(chr(c) for c in list(range(0x200b, 0x200f + 1)) +
                                           list(range(0x202a, 0x202e + 1)) +
                                           list(range(0x2060, 0x206f + 1)))
    for ch in invisible_chars:
        name = name.replace(ch, '')
    name = name.replace(' ', '').lower()
    return name

columns_map = {
    '상병코드': '상병코드',
    '수술코드': '상병코드',
    '코드': '상병코드',

    '상병명': '상병명',
    '상    병    명': '상병명',
    '상  병  명': '상병명',
    ' 상병명 ': '상병명',
    '수술 및 처치명': '상병명',
    '명 ': '상병명',

    '국비': '국비',
    ' 국비 ': '국비',
    '사비': '사비',
    ' 사비 ': '사비',
    '사비 ': '사비',

    '계': '합계',
    '합계': '합계',

    '구분': '구분',
    'Unnamed: 4': None,
    '년도': '년도',
    '지역': '지역',
}

normalized_columns_map = {}
for original_name, standard_name in columns_map.items():
    norm_name = normalize_column_name(original_name)
    if norm_name not in normalized_columns_map:
        normalized_columns_map[norm_name] = standard_name

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
    if '번호' in df.columns:
        df = df.drop(columns=['번호'])

    df = rename_columns(df, normalized_columns_map)
    df = strip_string_values(df)

    if None in df.columns:
        df = df.drop(columns=[None])

    if '국비' in df.columns:
        df['국비'] = pd.to_numeric(df['국비'], errors='coerce').fillna(0)
    if '사비' in df.columns:
        df['사비'] = pd.to_numeric(df['사비'], errors='coerce').fillna(0)
    if '국비' in df.columns and '사비' in df.columns:
        df['합계'] = df['국비'] + df['사비']

    dfs[i] = df
    print(f"[{i}] 열 이름: {df.columns.tolist()}")

merged_df = pd.concat(dfs, ignore_index=True)
merged_df.to_csv('C:/Users/julia/Downloads/질병 및 수술 통계.csv', index=False, encoding='UTF-8')