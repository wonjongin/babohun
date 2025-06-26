import pandas as pd

# 데이터 불러오기
df = pd.read_csv('final_merged_data/연령대별 진료과 예약건수_2023.csv')
df_code = pd.read_csv('map_merged_data/질병 및 수술 통계_코드통합.csv')

# (국비/합계) 비율 컬럼 추가
df_code['합계'] = df_code['합계'].replace(0, pd.NA)
df_code['국비비율'] = df_code['국비'] / df_code['합계']

# 평균 가중치 계산
mean_weight = df_code['국비비율'].mean(skipna=True)

print(f'국비/합계 평균 가중치: {mean_weight:.4f}')

# 곱할 열 목록
cols_to_multiply = ['20대', '30대', '40대', '50대', '60대', '70대', '80대', '90대']

# 각 열에 평균 가중치 곱해서 대치
for col in cols_to_multiply:
    df[col] = (pd.to_numeric(df[col], errors='coerce') * mean_weight).round(2)

# 결과 저장 (소수점 2자리로)
df.to_csv('map_merged_data/연령대별_진료과_예약건수_가중치.csv', index=False, float_format='%.2f')
