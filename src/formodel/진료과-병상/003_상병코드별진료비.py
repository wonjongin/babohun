import pandas as pd

# 데이터 불러오기
df = pd.read_csv('data/입원일자/건강보험심사평가원_3단상병별 성별 입원외래별 통계_20231231.csv')

# 숫자형 컬럼 공백 제거 및 변환
for col in ['환자수', '보험자부담금(선별포함)', '요양급여비용총액(선별포함)']:
    df[col] = df[col].astype(str).str.replace(',', '').str.strip().astype(float)

# 1. 남녀 합산, 입원/외래 구분별 평균진료비
agg1 = df.groupby(['주상병코드', '입원외래구분'], as_index=False).agg({
    '환자수': 'sum',
    '보험자부담금(선별포함)': 'sum',
    '요양급여비용총액(선별포함)': 'sum'
})
agg1['평균보험자부담금'] = agg1['보험자부담금(선별포함)'] / agg1['환자수']
agg1['평균요양급여비용총액'] = agg1['요양급여비용총액(선별포함)'] / agg1['환자수']

# 2. 남녀 합산, 입원+외래 전체 평균진료비
agg2 = df.groupby('주상병코드', as_index=False).agg({
    '환자수': 'sum',
    '보험자부담금(선별포함)': 'sum',
    '요양급여비용총액(선별포함)': 'sum'
})
agg2['평균보험자부담금'] = agg2['보험자부담금(선별포함)'] / agg2['환자수']
agg2['평균요양급여비용총액'] = agg2['요양급여비용총액(선별포함)'] / agg2['환자수']

# 샘플 출력
print('입원/외래 구분별 평균진료비:')
print(agg1.head())
print('\n전체(입원+외래) 평균진료비:')
print(agg2.head())

# 저장(선택)
agg1.to_csv('new_merged_data/상병코드별_입원외래구분_평균진료비.csv', index=False)
agg2.to_csv('new_merged_data/상병코드별_전체_평균진료비.csv', index=False)
