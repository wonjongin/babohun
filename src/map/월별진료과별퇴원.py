import pandas as pd

# 데이터 불러오기
df = pd.read_csv('final_merged_data/월별 진료과별 퇴원.csv')
df_code = pd.read_csv('map_merged_data/질병 및 수술 통계_코드통합.csv')

# (국비/합계) 비율 컬럼 추가
df_code['합계'] = df_code['합계'].replace(0, pd.NA)
df_code['국비비율'] = df_code['국비'] / df_code['합계']

# 평균 가중치 계산
mean_weight = df_code['국비비율'].mean(skipna=True)

# 주상병코드가 있는 경우 코드그룹 추출
df['코드그룹'] = df['주상병코드'].astype(str).str.strip().str[:3]

# 코드그룹, 년도, 지역 기준으로 가중치 병합
df = pd.merge(
    df,
    df_code[['코드그룹', '년도', '지역', '국비비율']],
    left_on=['코드그룹', '년도', '지역'],
    right_on=['코드그룹', '년도', '지역'],
    how='left'
)

# 건수 계산: 주상병코드가 있으면 해당 가중치, 없으면 평균 가중치
def apply_weight(row):
    if pd.notna(row['주상병코드']) and pd.notna(row['국비비율']):
        return round(row['건수'] * row['국비비율'], 2)
    else:
        return round(row['건수'] * mean_weight, 2)

df['건수'] = df.apply(apply_weight, axis=1)

# 필요시 코드그룹, 국비비율 컬럼 제거
df = df.drop(columns=['코드그룹', '국비비율'])

# 결과 저장
df.to_csv('map_merged_data/월별_진료과별_퇴원_가중치.csv', index=False, float_format='%.2f')
