import os
import glob
import pandas as pd
import re
import unicodedata
import csv

# 데이터 폴더 경로
DATA_DIR = 'data/연도별 과별 입원외래 환자/입원'
SAVE_PATH = 'new_merged_data/연도별진료과별입원환자수.csv'

result = []
csv_files = glob.glob(os.path.join(DATA_DIR, '*.csv'))

def is_float(val):
    try:
        return '.' in str(val) and float(val)
    except:
        return False

def simple_csv_to_df(filepath, skiprows=0, encoding='utf-8'):
    """
    csv 파일을 직접 읽어서 skiprows만큼 윗줄을 건너뛰고,
    남은 줄을 DataFrame으로 반환 (패딩 없음, 필드 개수 불일치 허용 안 함)
    """
    rows = []
    with open(filepath, encoding=encoding) as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i < skiprows:
                continue
            rows.append(row)
    df = pd.DataFrame(rows)
    return df

for file in csv_files:
    basename = os.path.basename(file)
    basename = unicodedata.normalize('NFC', basename)
    m = re.match(r'(\d{4})([가-힣]+).*입원.*\.csv', basename)
    if m:
        year = m.group(1)
        hospital = m.group(2)
    else:
        print(f'파일명에서 연도/병원 추출 실패: {basename}')
        continue  # 추출 실패 파일은 건너뜀

    # 연도에 따라 헤더 줄수 결정
    try:
        header_row = 1 if int(year) >= 2018 else 0
    except:
        header_row = 0

    # simple_csv_to_df로 파일 읽기
    try:
        df = simple_csv_to_df(file, skiprows=header_row)
    except Exception as e:
        print(f'헤더-데이터 열 개수 불일치 등 에러 발생, fallback: {basename}, 에러: {e}')
        continue

    dept_col = 0  # 항상 0번 컬럼에서 진료과명 추출

    for _, row in df.iterrows():
        dept = str(row.iloc[dept_col]).strip().replace('"', '')
        # 한글/영문 시작 진료과명만 허용
        if not re.match(r'^[A-Za-z가-힣]', dept):
            continue
        if dept in ['합계', '합계계', '계', '소계', '총계', 'nan', '-', '', '신생아', 'NB', 'NP', '구분', '일반']:
            continue
        # 마지막 열, 마지막-1 열 값 확인
        last_val = row.iloc[-1]
        second_last_val = row.iloc[-2] if len(row) > 1 else ''
        # 마지막 열이 소수점이면 일평균 환자, 아니면 합계
        if is_float(last_val):
            total = second_last_val
        else:
            total = last_val
        total = re.sub(r'[^0-9]', '', str(total))
        if total == '':
            total = '0'
        result.append([year, hospital, dept, total])

result_df = pd.DataFrame(result, columns=['연도', '병원', '진료과', '전체환자수'])

# ===== 후처리 시작 =====
# 진료과명 공백 제거
result_df['진료과'] = result_df['진료과'].str.replace(' ', '', regex=False)

# 전체환자수 빈값, nan, None을 모두 '0'으로 대체
result_df['전체환자수'] = result_df['전체환자수'].replace(['', None, 'nan', pd.NA], '0')
result_df['전체환자수'] = result_df['전체환자수'].fillna('0')

# 합계, 신생아 등 불필요 행 제거 (진료과명 기준)
불필요_진료과 = ['합계', '합계계', '계', '소계', '총계', '신생아', 'NB', 'NP', '신생아실', '합계계', '합계', '신생아', '신생아실']
result_df = result_df[~result_df['진료과'].isin(불필요_진료과)]

# 진료과명 영어 약어를 한글로 변환
replace_dict = {
    "IM": "내과",
    "NE": "신경과",
    "PED": "소아청소년과",
    "GS": "외과",
    "CS": "정신건강의학과",      # 2015년에만 있고 2016년부터는 진료 없음
    "OS": "정형외과",
    "NS": "신경외과",
    "OG": "산부인과",
    "OPH": "안과",
    "DM": "피부과",            # 추정 (Dermatology)
    "ENT": "이비인후과",
    "URO": "비뇨의학과",
    "DT": "치과",              # 추정 (Dentistry)
    "RM": "재활의학과",         # 또는 재활센터
    "FM": "가정의학과",
    "ER": "응급의학과",
    "NM": "핵의학과"           # Nuclear Medicine
}
result_df['진료과'] = result_df['진료과'].replace(replace_dict)

# ===== 후처리 끝 =====

print('입원 데이터 샘플:')
print(result_df.head(10))

os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
result_df.to_csv(SAVE_PATH, index=False, encoding='utf-8')

print(f'저장 완료: {SAVE_PATH}')
