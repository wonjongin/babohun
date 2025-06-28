import pandas as pd

# 데이터 불러오기
df = pd.read_csv('new_merged_data/상병코드별_진료과_건수_더미_병합.csv')

# 진료과별 건수만 추출 (상명코드 제외)
진료과_컬럼 = df.columns[1:]
df['최다진료과'] = df[진료과_컬럼].idxmax(axis=1)

# 상명코드와 최다진료과만 남기기
result = df[['상명코드', '최다진료과']]

# 결과 저장
result.to_csv('new_merged_data/상병코드별_최다진료과.csv', index=False, encoding='utf-8')
