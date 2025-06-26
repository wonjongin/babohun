import os
import glob
import pandas as pd
import re
import unicodedata
import csv
import pandas as pd

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


# 데이터 폴더 경로 (외래)
DATA_DIR = 'data/연도별 과별 입원외래 환자/외래'
SAVE_PATH = 'new_merged_data/연도별진료과별외래환자수.csv'

result = []
csv_files = glob.glob(os.path.join(DATA_DIR, '*.csv'))

def is_float(val):
    try:
        return '.' in str(val) and float(val)
    except:
        return False

for file in csv_files:
    basename = os.path.basename(file)
    basename = unicodedata.normalize('NFC', basename)
    m = re.match(r'(\d{4})([가-힣]+).*외래.*\.csv', basename)
    if m:
        year = m.group(1)
        hospital = m.group(2)
    else:
        print(f'파일명에서 연도/병원 추출 실패: {basename}')
        year, hospital = basename, basename

    # 규칙에 따라 헤더 줄수, 합계 위치 결정
    # try:
    year_int = int(year)

    # except:
    #     year_int = 2017
    # 기본값
    header_row = 0
    total_col = -1
    if year_int <= 2017:
        header_row = 0
        # 합계/일평균환자 구분(입원과 동일)
        total_col = None  # None이면 아래에서 is_float로 판별
    else:
        if '월별' in basename:
            if basename.endswith('3.csv'):
                header_row = 2
            else:
                header_row = 1
            total_col = -1
        elif '환자유형별' in basename:
            header_row = 2
            total_col = -4
        else:
            header_row = 1
            total_col = -1
    # 헤더-데이터 열 개수 불일치 대비
    try:
        df = simple_csv_to_df(file, header_row)
        # df = pd.read_csv(file, header=None, dtype=str, skiprows=header_row, engine='python')
    except Exception as e:
        print(f'헤더-데이터 열 개수 불일치 등 에러 발생, fallback: {basename}, 에러: {e}')
        df = pd.read_csv(file, header=None, dtype=str)
        df.columns = [f'col{i}' for i in range(len(df.columns))]

    for _, row in df.iterrows():
        dept = str(row.iloc[0]).strip().replace('"', '')
        # 한글/영문 시작 진료과명만 허용
        # if not re.match(r'^[A-Za-z가-힣]', dept):
        #     continue
        if dept in ['합계', '합계계', '계', '소계', '총계', 'nan', '-', '', '신생아', 'NB', 'NP', '초진', '재진', '구분']:
            continue
        # 합계 위치 규칙 적용
        if total_col is None:
            # 2014~2017: 마지막 열이 소수점이면 일평균환자, 아니면 합계
            last_val = row.iloc[-1]
            second_last_val = row.iloc[-2] if len(row) > 1 else ''
            if is_float(last_val):
                total = second_last_val
            else:
                total = last_val
        else:
            total = row.iloc[total_col]
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
    "DE": "피부과",
    "AD": "알레르기내과",     # 또는 '기타'로 처리 가능
    "HC": "건강관리과",
    "KA": "한의과",
    "ENT": "이비인후과",
    "URO": "비뇨의학과",
    "DT": "치과",              # 추정 (Dentistry)
    "RM": "재활의학과",         # 또는 재활센터
    "FM": "가정의학과",
    "ER": "응급의학과",
    "NM": "핵의학과"           # Nuclear Medicine
}
result_df['진료과'] = result_df['진료과'].replace(replace_dict)

# 진료과명에서 줄바꿈, 탭, 공백을 모두 제거한 후 '기타'가 포함되어 있으면 모두 '기타'로 통일
result_df['진료과'] = result_df['진료과'].apply(lambda x: '기타' if '기타' in x.replace(' ', '').replace('\n', '').replace('\t', '') else x)

# 합계, 신생아 등 불필요 행 제거 (진료과명 기준)
불필요_진료과 = ['합계', '합계계', '계', '소계', '총계', '신생아', 'NB', 'NP', '신생아실', '합계계', '합계', '신생아', '신생아실']
result_df = result_df[~result_df['진료과'].isin(불필요_진료과)]

# ===== 후처리 끝 =====

print('외래 데이터 샘플:')
print(result_df.head(10))

os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
result_df.to_csv(SAVE_PATH, index=False, encoding='utf-8')

print(f'저장 완료: {SAVE_PATH}')
