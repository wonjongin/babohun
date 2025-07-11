import pandas as pd

# 1. 2023년 병원별 전체 환자수 집계
df_patient = pd.read_csv('new_merged_data/연도별진료과별_입원외래_통합환자수.csv', encoding='utf-8-sig')
df_2023 = df_patient[df_patient['연도'] == 2023]
patient_sum = df_2023.groupby('병원')['전체환자수_합계'].sum().reset_index()
patient_sum.columns = ['병원', '내원환자수']

# 2. 병원별 일반병상수, 전문의수 추출
df_resource = pd.read_csv('new_merged_data/병원_통합_데이터_호스피스 삭제.csv', encoding='utf-8-sig')

# 병원명 매핑 (중앙=서울)
hospital_map = {
    '중앙': '서울',
    '서울': '서울',
    '부산': '부산',
    '대전': '대전',
    '대구': '대구',
    '광주': '광주',
    '인천': '인천',
}

# 병원명 표준화
def std_hospital_name(name):
    for k, v in hospital_map.items():
        if k in name:
            return v
    return name

df_resource['병원'] = df_resource['병원명'].apply(std_hospital_name)

# 일반병상수 = 일반입원실_상급 + 일반입원실_일반
bed_cols = ['일반입원실_상급', '일반입원실_일반']
df_resource['일반병상수'] = df_resource[bed_cols].sum(axis=1)

# 전문의수 = '_전문의수'로 끝나는 모든 컬럼 합계
specialist_cols = [col for col in df_resource.columns if col.endswith('_전문의수')]
df_resource['전문의수'] = df_resource[specialist_cols].sum(axis=1)

# 병원별 자원 데이터 추출
df_resource_summary = df_resource[['병원', '일반병상수', '전문의수']]

# 3. 병원명 매핑(중앙=서울) 반영해서 환자수 병원명도 맞추기
def std_patient_hospital(name):
    if name == '중앙':
        return '서울'
    return name

patient_sum['병원'] = patient_sum['병원'].apply(std_patient_hospital)

# 4. 병원별 데이터 병합
df_merged = pd.merge(patient_sum, df_resource_summary, on='병원', how='inner')

# 5. 지표 계산
# (1) 내원환자수 대비 병상수 (환자수/병상수)
df_merged['환자수_대비_병상수'] = df_merged['내원환자수'] / df_merged['일반병상수']
# (2) 내원환자수 대비 전문의수 (환자수/전문의수)
df_merged['환자수_대비_전문의수'] = df_merged['내원환자수'] / df_merged['전문의수']

# 6. 컬럼 순서 정리 및 저장
cols = ['병원', '내원환자수', '일반병상수', '전문의수', '환자수_대비_병상수', '환자수_대비_전문의수']
df_merged = df_merged[cols]

# 7. 결과 저장
save_path = 'results/병원별_내원환자_대비_자원지표.csv'
df_merged.to_csv(save_path, index=False, encoding='utf-8-sig')
print(f'✅ 저장 완료: {save_path}')
print(df_merged)
