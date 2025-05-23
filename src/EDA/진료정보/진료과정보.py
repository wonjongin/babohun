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
    
departmentInfo_Gwangju_2019 = process_file('광주', 2019, "data/진료정보/보훈병원_진료과정보_광주보훈병원_2019.csv")
departmentInfo_Gwangju_2020 = process_file('광주', 2020, "data/진료정보/보훈병원_진료과정보_광주보훈병원_2020.csv")
departmentInfo_Gwangju_2021 = process_file('광주', 2021, "data/진료정보/보훈병원_진료과정보_광주보훈병원_2021.csv")
departmentInfo_Gwangju_2022 = process_file('광주', 2022, "data/진료정보/보훈병원_진료과정보_광주보훈병원_2022.csv")
departmentInfo_Gwangju_2023 = process_file('광주', 2023, "data/진료정보/보훈병원_진료과정보_광주보훈병원_2023.csv")

departmentInfo_Daegu_2019 = process_file('대구', 2019, "data/진료정보/보훈병원_진료과정보_대구보훈병원_2019.csv")
departmentInfo_Daegu_2020 = process_file('대구', 2020, "data/진료정보/보훈병원_진료과정보_대구보훈병원_2020.csv")
departmentInfo_Daegu_2021 = process_file('대구', 2021, "data/진료정보/보훈병원_진료과정보_대구보훈병원_2021.csv")
departmentInfo_Daegu_2022 = process_file('대구', 2022, "data/진료정보/보훈병원_진료과정보_대구보훈병원_2022.csv")
departmentInfo_Daegu_2023 = process_file('대구', 2023, "data/진료정보/보훈병원_진료과정보_대구보훈병원_2023.csv")

departmentInfo_Daejeon_2019 = process_file('대전', 2019, "data/진료정보/보훈병원_진료과정보_대전보훈병원_2019.csv")
departmentInfo_Daejeon_2020 = process_file('대전', 2020, "data/진료정보/보훈병원_진료과정보_대전보훈병원_2020.csv")
departmentInfo_Daejeon_2021 = process_file('대전', 2021, "data/진료정보/보훈병원_진료과정보_대전보훈병원_2021.csv")
departmentInfo_Daejeon_2022 = process_file('대전', 2022, "data/진료정보/보훈병원_진료과정보_대전보훈병원_2022.csv")
departmentInfo_Daejeon_2023 = process_file('대전', 2023, "data/진료정보/보훈병원_진료과정보_대전보훈병원_2023.csv")

departmentInfo_Busan_2019 = process_file('부산', 2019, "data/진료정보/보훈병원_진료과정보_부산보훈병원_2019.csv")
departmentInfo_Busan_2020 = process_file('부산', 2020, "data/진료정보/보훈병원_진료과정보_부산보훈병원_2020.csv")
departmentInfo_Busan_2021 = process_file('부산', 2021, "data/진료정보/보훈병원_진료과정보_부산보훈병원_2021.csv")
departmentInfo_Busan_2022 = process_file('부산', 2022, "data/진료정보/보훈병원_진료과정보_부산보훈병원_2022.csv")
departmentInfo_Busan_2023 = process_file('부산', 2023, "data/진료정보/보훈병원_진료과정보_부산보훈병원_2023.csv")

departmentInfo_Incheon_2019 = process_file('인천', 2019, "data/진료정보/보훈병원_진료과정보_인천보훈병원_2019.csv")
departmentInfo_Incheon_2020 = process_file('인천', 2020, "data/진료정보/보훈병원_진료과정보_인천보훈병원_2020.csv")
departmentInfo_Incheon_2021 = process_file('인천', 2021, "data/진료정보/보훈병원_진료과정보_인천보훈병원_2021.csv")
departmentInfo_Incheon_2022 = process_file('인천', 2022, "data/진료정보/보훈병원_진료과정보_인천보훈병원_2022.csv")
departmentInfo_Incheon_2023 = process_file('인천', 2023, "data/진료정보/보훈병원_진료과정보_인천보훈병원_2023.csv")

departmentInfo_Seoul_2019 = process_file('서울', 2019, "data/진료정보/보훈병원_진료과정보_중앙보훈병원_2019.csv")
departmentInfo_Seoul_2020 = process_file('서울', 2020, "data/진료정보/보훈병원_진료과정보_중앙보훈병원_2020.csv")
departmentInfo_Seoul_2021 = process_file('서울', 2021, "data/진료정보/보훈병원_진료과정보_중앙보훈병원_2021.csv")
departmentInfo_Seoul_2022 = process_file('서울', 2022, "data/진료정보/보훈병원_진료과정보_중앙보훈병원_2022.csv")
departmentInfo_Seoul_2023 = process_file('서울', 2023, "data/진료정보/보훈병원_진료과정보_중앙보훈병원_2023.csv")

# 열 이름 통합 #
dfs = [
    departmentInfo_Busan_2019, departmentInfo_Busan_2020, departmentInfo_Busan_2021, departmentInfo_Busan_2022, departmentInfo_Busan_2023,
    departmentInfo_Daegu_2019, departmentInfo_Daegu_2020, departmentInfo_Daegu_2021, departmentInfo_Daegu_2022, departmentInfo_Daegu_2023, 
    departmentInfo_Daejeon_2019, departmentInfo_Daejeon_2020, departmentInfo_Daejeon_2021, departmentInfo_Daejeon_2022, departmentInfo_Daejeon_2023,
    departmentInfo_Busan_2019, departmentInfo_Busan_2020, departmentInfo_Busan_2021, departmentInfo_Busan_2022, departmentInfo_Busan_2023,
    departmentInfo_Gwangju_2019, departmentInfo_Gwangju_2020, departmentInfo_Gwangju_2021, departmentInfo_Gwangju_2022, departmentInfo_Gwangju_2023,
    departmentInfo_Incheon_2019, departmentInfo_Incheon_2020, departmentInfo_Incheon_2021, departmentInfo_Incheon_2022, departmentInfo_Incheon_2023,
    departmentInfo_Seoul_2019, departmentInfo_Seoul_2020, departmentInfo_Seoul_2021, departmentInfo_Seoul_2022, departmentInfo_Seoul_2023
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
    # 진료과/진료과별
    '진료과': '진료과별',
    '진 료 과 별': '진료과별',
    '진료과별': '진료과별',

    # 소속부서/부속부서
    '소속부서': '부속부서',
    '소 속 부 서': '부속부서',
    '부속부서': '부속부서',

    # 보유장비
    '보유장비': '보유장비',
    '보 유 장 비(한글)': '보유장비',
    '보유장비(한글)': '보유장비',

    # 진료내용
    '진료내용': '진료내용',
    '진     료     내     용': '진료내용',
    '진  료  내  용': '진료내용',
    '진료 내용': '진료내용',
    '내                     용': '진료내용',
    '내용': '진료내용',

    # 년도, 지역
    '년도': '년도',
    '지역': '지역',
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
    if '번호' in df.columns:
        df = df.drop(columns=['번호'])

    df = rename_columns(df, normalized_columns_map)
    df = strip_string_values(df)

    dfs[i] = df
    print(f"[{i}] 열 이름: {df.columns.tolist()}")

merged_df = pd.concat(dfs, ignore_index=True)

values = merged_df['진료과별'].dropna().unique()
values_sorted = sorted(values)

for v in values_sorted:
    print(v)

merged_df = merged_df[~merged_df['진료과별'].astype(str).str.contains('안센터')]

mapping = {
    # 내과 계열
    '내     과': '내과',
    '내과': '내과',
    '내분비내과': '내분비내과',
    '류마티스내과': '류마티스내과',
    '소화기내과': '소화기내과',
    '순환기내과': '순환기내과',
    '신장내과': '신장내과',
    '혈액종양내과': '혈액종양내과',
    '호흡기내과': '호흡기내과',
    '감염내과': '감염내과',

    # 외과 계열
    '외과': '외과',
    '일반외과': '외과',
    '흉부외과': '흉부외과',
    '신경외과': '신경외과',

    # 소아/산부인과 등
    '산부인과': '산부인과',
    '소아청소년과': '소아청소년과',

    # 기타 진료과
    '가정의학과': '가정의학과',
    '건강검진과': '건강검진과',
    '건강관리과': '건강관리과',
    '마취통증의학과': '마취통증의학과',
    '방사선종양학과': '방사선종양학과',
    '병리과': '병리과',
    '비뇨기과': '비뇨의학과',
    '비뇨의학과': '비뇨의학과',
    '성형외과': '성형외과',
    '신 경 과': '신경과',
    '신경과': '신경과',
    '영상의학과': '영상의학과',
    '이비인후과': '이비인후과',
    '재활의학과': '재활의학과',
    '재활의학과(재활센터)': '재활의학과',
    '재활센터': '재활의학과',
    '정신건강의학과': '정신건강의학과',
    '정형외과': '정형외과',
    '진단,검사의학과': '진단검사의학과',
    '진단검사의학과': '진단검사의학과',
    '핵의학과': '핵의학과',
    '피 부 과': '피부과',
    '피부과': '피부과',

    # 치과 계열
    '치     과': '치과',
    '치  과': '치과',
    '치 과': '치과',
    '치과': '치과',
    '치과병원': '치과',

    # 한방 계열
    '한방과': '한방과',
    '한방진료과': '한방과',
    '한의과': '한방과',

    # 안과 계열
    '안     과': '안과',
    '안  과': '안과',
    '안과': '안과',

    # 센터/실/병동 등
    '암센터': '암센터',
    '욕창센터': '욕창센터',
    '중환자실': '중환자실',
    '응급실': '응급의학과',
    '응급의학과': '응급의학과',
    '응급의학과(지역응급의료센터)': '응급의학과',
}

merged_df['진료과별'] = merged_df['진료과별'].astype(str).str.strip()
merged_df['진료과별'] = merged_df['진료과별'].replace(mapping) 

merged_df.to_csv('C:/Users/julia/Downloads/진료과정보.csv', index=False, encoding='UTF-8')