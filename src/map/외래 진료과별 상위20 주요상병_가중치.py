import pandas as pd

# 데이터 불러오기
df = pd.read_csv('final_merged_data/외래 진료과별 상위20 주요상병.csv')
df_code = pd.read_csv('map_merged_data/질병 및 수술 통계_코드통합.csv')

# 코드그룹 추출 (상명코드 앞 3글자, 공백 제거)
df['코드그룹'] = df['상명코드'].astype(str).str.strip().str[:3]

# 코드통합에서 '합계'가 0인 경우 NA로 처리
df_code['합계'] = df_code['합계'].replace(0, pd.NA)

# (국비/합계) 비율 컬럼 추가
df_code['국비비율'] = df_code['국비'] / df_code['합계']

# 병합: 코드그룹, 년도, 지역 기준
merged = pd.merge(
    df,
    df_code[['코드그룹', '년도', '지역', '국비비율']],
    left_on=['코드그룹', '년도', '지역'],
    right_on=['코드그룹', '년도', '지역'],
    how='left'
)

# 건수에 가중치 곱해서 대치
merged['건수'] = (merged['건수'] * merged['국비비율']).round(2)

# 필요시 코드그룹, 국비비율 컬럼 제거
# merged = merged.drop(columns=['코드그룹', '국비비율'])

# 결과 저장 (소수점 2자리로)
merged.to_csv('map_merged_data/외래_진료과별_상위20_주요상병_가중치.csv', index=False)
