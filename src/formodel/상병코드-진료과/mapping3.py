import pandas as pd
import numpy as np
import os

# --------------------------------------------------
# 1) 데이터 로드
# --------------------------------------------------
print("=== 데이터 로드 시작 ===")

# 상병코드별 최다진료과 파일 로드
print("1/4: 상병코드별_최다진료과.csv 로드 중...")
most_dept_df = pd.read_csv("new_merged_data/상병코드별_최다진료과.csv")
print(f"  - 로드된 상병코드 수: {len(most_dept_df)}")
print(f"  - 컬럼: {list(most_dept_df.columns)}")
print(f"  - 샘플 데이터:")
print(most_dept_df.head())

# 심평원 자료 로드
print("\n2/4: 심평원 자료 로드 중...")
hira_df = pd.read_csv("data/상병코드 진료과/건강보험심사평가원_병원급이상 진료과목별 상병 통계_20231231.csv")
print(f"  - 로드된 데이터 크기: {hira_df.shape}")
print(f"  - 컬럼: {list(hira_df.columns)}")
print(f"  - 샘플 데이터:")
print(hira_df.head())

# --------------------------------------------------
# 2) 데이터 전처리
# --------------------------------------------------
print("\n=== 데이터 전처리 시작 ===")

# 심평원 자료 컬럼명 정리
print("3/4: 심평원 자료 전처리 중...")
hira_df.columns = ['진료년도', '진료과목', '주상병코드', '환자수', '명세서청구건수', '입내원일수', '보험자부담금', '요양급여비용총액']

# 데이터 타입 확인
print(f"  - 환자수 컬럼 타입: {hira_df['환자수'].dtype}")
print(f"  - 환자수 샘플 값: {hira_df['환자수'].head().tolist()}")

# 환자수를 숫자로 변환 (문자열인 경우에만 쉼표 제거)
if hira_df['환자수'].dtype == 'object':
    hira_df['환자수'] = pd.to_numeric(hira_df['환자수'].str.replace(',', ''), errors='coerce')
    hira_df['명세서청구건수'] = pd.to_numeric(hira_df['명세서청구건수'].str.replace(',', ''), errors='coerce')
    hira_df['입내원일수'] = pd.to_numeric(hira_df['입내원일수'].str.replace(',', ''), errors='coerce')
else:
    # 이미 숫자형인 경우 그대로 사용
    hira_df['환자수'] = pd.to_numeric(hira_df['환자수'], errors='coerce')
    hira_df['명세서청구건수'] = pd.to_numeric(hira_df['명세서청구건수'], errors='coerce')
    hira_df['입내원일수'] = pd.to_numeric(hira_df['입내원일수'], errors='coerce')

# 결측값 처리
hira_df = hira_df.dropna(subset=['환자수', '주상병코드', '진료과목'])
print(f"  - 전처리 후 데이터 크기: {hira_df.shape}")

# 상병코드 정리 (앞뒤 공백 제거)
hira_df['주상병코드'] = hira_df['주상병코드'].str.strip()
most_dept_df['상명코드'] = most_dept_df['상명코드'].str.strip()

print(f"  - 심평원 자료 상병코드 샘플: {hira_df['주상병코드'].unique()[:10]}")
print(f"  - 최다진료과 자료 상병코드 샘플: {most_dept_df['상명코드'].unique()[:10]}")

# --------------------------------------------------
# 3) 환자수 기준 최다진료과 찾기
# --------------------------------------------------
print("\n4/4: 환자수 기준 최다진료과 찾기 중...")

# 상병코드별로 환자수가 가장 많은 진료과 찾기
patient_count_by_dept = hira_df.groupby(['주상병코드', '진료과목'])['환자수'].sum().reset_index()
max_patient_dept = patient_count_by_dept.loc[patient_count_by_dept.groupby('주상병코드')['환자수'].idxmax()]

print(f"  - 환자수 기준 최다진료과 찾은 상병코드 수: {len(max_patient_dept)}")
print(f"  - 샘플 결과:")
print(max_patient_dept.head())

# --------------------------------------------------
# 4) 매핑 생성
# --------------------------------------------------
print("\n=== 매핑 생성 시작 ===")

# 기존 최다진료과 매핑을 기준으로 하고, 심평원 자료로 보완
mapping_dict = {}

# 1단계: 기존 최다진료과 매핑 사용
print("1/3: 기존 최다진료과 매핑 적용 중...")
for _, row in most_dept_df.iterrows():
    disease_code = row['상명코드']
    dept = row['최다진료과']
    mapping_dict[disease_code] = {
        '상병코드': disease_code,
        '진료과': dept,
        '매핑_근거': '기존_최다진료과',
        '환자수': None,
        '진료과목_수': None
    }

print(f"  - 기존 매핑 적용된 상병코드 수: {len(mapping_dict)}")

# 2단계: 심평원 자료에서 누락된 상병코드 보완
print("2/3: 심평원 자료로 누락된 상병코드 보완 중...")
hira_disease_codes = set(hira_df['주상병코드'].unique())
existing_codes = set(mapping_dict.keys())
missing_codes = hira_disease_codes - existing_codes

print(f"  - 심평원 자료 총 상병코드 수: {len(hira_disease_codes)}")
print(f"  - 기존 매핑에 있는 상병코드 수: {len(existing_codes)}")
print(f"  - 누락된 상병코드 수: {len(missing_codes)}")

# 누락된 상병코드에 대해 환자수 기준 최다진료과 적용
for disease_code in missing_codes:
    if disease_code in max_patient_dept['주상병코드'].values:
        dept_info = max_patient_dept[max_patient_dept['주상병코드'] == disease_code].iloc[0]
        mapping_dict[disease_code] = {
            '상병코드': disease_code,
            '진료과': dept_info['진료과목'],
            '매핑_근거': '환자수_최다진료과',
            '환자수': dept_info['환자수'],
            '진료과목_수': len(hira_df[hira_df['주상병코드'] == disease_code])
        }

print(f"  - 환자수 기준으로 추가된 상병코드 수: {len(missing_codes)}")

# 3단계: 기존 매핑의 환자수 정보도 업데이트
print("3/3: 기존 매핑의 환자수 정보 업데이트 중...")
for disease_code in existing_codes:
    if disease_code in max_patient_dept['주상병코드'].values:
        dept_info = max_patient_dept[max_patient_dept['주상병코드'] == disease_code].iloc[0]
        mapping_dict[disease_code]['환자수'] = dept_info['환자수']
        mapping_dict[disease_code]['진료과목_수'] = len(hira_df[hira_df['주상병코드'] == disease_code])

# --------------------------------------------------
# 5) 결과 데이터프레임 생성
# --------------------------------------------------
print("\n=== 결과 데이터프레임 생성 ===")

# 딕셔너리를 데이터프레임으로 변환
mapping_df = pd.DataFrame(list(mapping_dict.values()))

# 컬럼 순서 정리
mapping_df = mapping_df[['상병코드', '진료과', '매핑_근거', '환자수', '진료과목_수']]

print(f"✅ 최종 매핑 데이터 크기: {mapping_df.shape}")
print(f"✅ 매핑 근거별 분포:")
print(mapping_df['매핑_근거'].value_counts())

print(f"\n✅ 진료과별 분포:")
print(mapping_df['진료과'].value_counts().head(10))

print(f"\n✅ 샘플 데이터:")
print(mapping_df.head(10))

# --------------------------------------------------
# 6) 결과 저장
# --------------------------------------------------
print("\n=== 결과 저장 ===")

# 결과 디렉토리 생성
output_dir = "new_merged_data"
os.makedirs(output_dir, exist_ok=True)

# CSV 파일로 저장
output_file = f"{output_dir}/상병코드_진료과_매핑3.csv"
mapping_df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"✅ 매핑 결과 저장 완료: {output_file}")

# --------------------------------------------------
# 7) 통계 요약
# --------------------------------------------------
print("\n=== 통계 요약 ===")

print(f"📊 전체 상병코드 수: {len(mapping_df)}")
print(f"📊 매핑 근거별 분포:")
for basis, count in mapping_df['매핑_근거'].value_counts().items():
    print(f"  - {basis}: {count}개 ({count/len(mapping_df)*100:.1f}%)")

print(f"\n📊 진료과별 분포 (상위 10개):")
for dept, count in mapping_df['진료과'].value_counts().head(10).items():
    print(f"  - {dept}: {count}개")

print(f"\n📊 환자수 통계:")
if mapping_df['환자수'].notna().any():
    print(f"  - 평균 환자수: {mapping_df['환자수'].mean():.0f}명")
    print(f"  - 중앙값 환자수: {mapping_df['환자수'].median():.0f}명")
    print(f"  - 최대 환자수: {mapping_df['환자수'].max():.0f}명")
    print(f"  - 최소 환자수: {mapping_df['환자수'].min():.0f}명")

print(f"\n📊 진료과목 수 통계:")
if mapping_df['진료과목_수'].notna().any():
    print(f"  - 평균 진료과목 수: {mapping_df['진료과목_수'].mean():.1f}개")
    print(f"  - 중앙값 진료과목 수: {mapping_df['진료과목_수'].median():.0f}개")
    print(f"  - 최대 진료과목 수: {mapping_df['진료과목_수'].max():.0f}개")

print(f"\n🎉 매핑3 생성 완료!")
print(f"📁 저장 위치: {output_file}")
