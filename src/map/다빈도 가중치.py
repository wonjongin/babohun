import pandas as pd

# 데이터 불러오기
df_freq = pd.read_csv('final_merged_data/다빈도 질환 환자 연령별 분포.csv')
df_code = pd.read_csv('map_merged_data/질병 및 수술 통계_코드통합.csv')

# 다빈도 질환의 상병코드에서 코드그룹 추출 (공백 제거 후 앞 3글자)
df_freq['코드그룹'] = df_freq['상병코드'].astype(str).str.strip().str[:3]

# 코드통합에서 '합계'가 0인 경우는 NA로 처리(division by zero 방지)
df_code['합계'] = df_code['합계'].replace(0, pd.NA)

# (국비/합계) 비율 컬럼 추가
df_code['국비비율'] = df_code['국비'] / df_code['합계']

# 병합: 코드그룹, 년도, 지역 기준
merged = pd.merge(
    df_freq,
    df_code[['코드그룹', '년도', '지역', '국비비율']],
    left_on=['코드그룹', '년도', '지역'],
    right_on=['코드그룹', '년도', '지역'],
    how='left'
)

# 곱할 열 목록
cols_to_multiply = ['실인원', '연인원', '59이하', '60-64', '65-69', '70-79', '80-89', '90이상', '진료비(천원)']

# 각 열에 곱해서 기존 열에 덮어쓰기
for col in cols_to_multiply:
    merged[col] = (merged[col] * merged['국비비율']).round(2)

# 코드그룹 컬럼 제거(필요시)
# merged = merged.drop(columns=['코드그룹', '국비비율'])

# 결과 저장
merged.to_csv('map_merged_data/다빈도_질환_국비비율_곱셈.csv', index=False)
