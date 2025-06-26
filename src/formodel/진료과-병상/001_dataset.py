import pandas as pd

# 데이터 불러오기
raw = pd.read_csv('new_merged_data/병원_통합_데이터.csv')

# 1. 호스피스 병원 제거
raw = raw[~raw['병원명'].str.contains('호스피스')].reset_index(drop=True)

# 2. 병상 관련 컬럼만 추출 ('병상' 또는 '실'이 포함된 컬럼)
bed_cols = [col for col in raw.columns if ('병상' in col or '실' in col)]

# 3. X, y 모두 병원명 포함
X = raw[['병원명'] + bed_cols]
y = raw[['병원명'] + bed_cols]

# 샘플 출력
print('X 샘플:')
print(X.head())
print('y 샘플:')
print(y.head())
