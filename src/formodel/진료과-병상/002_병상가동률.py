import pandas as pd

# 데이터 불러오기
df = pd.read_csv('data/입원일자/건강보험심사평가원_3단상병별 성별 입원외래별 통계_20231231.csv')

# 1. 입원만 필터링
df = df[df['입원외래구분'] == '입원']

# 2. 남녀 합산 (성별 무시)
# 3. 상병코드별 입원일수, 환자수 합계
agg = df.groupby('주상병코드', as_index=False).agg({'입내원일수': 'sum', '환자수': 'sum'})

# 4. 평균입원일수 계산
agg['평균입원일수'] = agg['입내원일수'] / agg['환자수']

# 5. 컬럼명 정리(선택)
agg = agg.rename(columns={'입내원일수': '총입원일수', '환자수': '총환자수'})

# 6. 샘플 출력
print(agg.head())

# 7. CSV로 저장
agg.to_csv('new_merged_data/상병코드별_평균입원일수.csv', index=False)
print('저장 완료: new_merged_data/상병코드별_평균입원일수.csv')
