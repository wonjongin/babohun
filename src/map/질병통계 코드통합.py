import pandas as pd

# 데이터 불러오기
df = pd.read_csv('final_merged_data/질병 및 수술 통계.csv')

# 질병코드 앞 3글자(알파벳+숫자2개)만 추출해서 새로운 컬럼 생성
df['코드그룹'] = df['상병코드'].str[:3]

# 그룹화 및 합계 집계
grouped = df.groupby(['코드그룹', '년도', '지역', '구분']).agg({
    '국비': 'sum',
    '사비': 'sum',
    '합계': 'sum'
}).reset_index()

# 정수형으로 변환
for col in ['국비', '사비', '합계']:
    grouped[col] = grouped[col].astype(int)

# 결과 저장
grouped.to_csv('map_merged_data/질병 및 수술 통계_코드통합.csv', index=False)
