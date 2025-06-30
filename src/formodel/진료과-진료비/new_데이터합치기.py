# -*- coding: utf-8 -*-
"""
Created on Sat Jun 28 20:43:03 2025
author: jenny

df_result2_with_심평원.csv에 진료비(천원) 컬럼 추가
"""

import pandas as pd
import numpy as np

# ----------------------------------------------------------------------
# 1) 데이터 로드
# ----------------------------------------------------------------------
print("=== 데이터 로딩 시작 ===")

# 매핑 데이터 (진료과 정보 포함)
mapping_csv = "new_merged_data/df_result2_with_심평원.csv"
# 진료비 데이터
cost_csv = "final_merged_data/다빈도 질환 환자 연령별 분포.csv"

# 데이터 로드
df_mapping = pd.read_csv(mapping_csv, encoding="utf-8-sig")
df_cost = pd.read_csv(cost_csv, encoding="utf-8-sig")

print(f"매핑 데이터 행 수: {len(df_mapping)}")
print(f"진료비 데이터 행 수: {len(df_cost)}")

# ----------------------------------------------------------------------
# 2) 데이터 전처리
# ----------------------------------------------------------------------
print("=== 데이터 전처리 ===")

# 진료비 데이터에서 필요한 컬럼만 선택
df_cost_subset = df_cost[['년도', '지역', '구분', '상병코드', '진료비(천원)']].copy()

# 상병코드 정리 (공백 제거)
df_cost_subset['상병코드'] = df_cost_subset['상병코드'].str.strip()
df_mapping['상병코드'] = df_mapping['상병코드'].str.strip()

print(f"진료비 데이터 고유 상병코드 수: {df_cost_subset['상병코드'].nunique()}")
print(f"매핑 데이터 고유 상병코드 수: {df_mapping['상병코드'].nunique()}")

# ----------------------------------------------------------------------
# 3) 진료비 데이터 매핑
# ----------------------------------------------------------------------
print("=== 진료비 데이터 매핑 ===")

# 매핑 전 상태 확인
print(f"매핑 전 df_mapping 행 수: {len(df_mapping)}")

# 진료비 데이터를 매핑 데이터에 merge
df_result = df_mapping.merge(
    df_cost_subset[['년도', '지역', '구분', '상병코드', '진료비(천원)']], 
    on=['년도', '지역', '구분', '상병코드'], 
    how='left'
)

print(f"매핑 후 df_result 행 수: {len(df_result)}")

# 매핑 성공률 확인
matched_count = df_result['진료비(천원)'].notna().sum()
match_rate = (matched_count / len(df_result)) * 100
print(f"진료비 매핑 성공: {matched_count}개 ({match_rate:.2f}%)")

# ----------------------------------------------------------------------
# 4) 매핑 실패 케이스 분석
# ----------------------------------------------------------------------
print("=== 매핑 실패 케이스 분석 ===")

# 매핑 실패한 행들 확인
failed_matches = df_result[df_result['진료비(천원)'].isna()]
print(f"매핑 실패한 행 수: {len(failed_matches)}")

if len(failed_matches) > 0:
    print("\n매핑 실패한 상병코드들:")
    failed_codes = failed_matches[['년도', '지역', '구분', '상병코드']].drop_duplicates()
    print(failed_codes.head(10))
    
    print(f"\n매핑 실패한 구분들:")
    print(failed_matches['구분'].value_counts())

# ----------------------------------------------------------------------
# 5) 결과 저장
# ----------------------------------------------------------------------
print("=== 결과 저장 ===")

# 결과 파일 저장
output_path = "new_merged_data/df_result2_with_심평원_진료비.csv"
df_result.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"결과가 '{output_path}'에 저장되었습니다!")
print(f"최종 데이터 행 수: {len(df_result)}")
print(f"진료비(천원) 컬럼 통계:")
print(f"  - 평균: {df_result['진료비(천원)'].mean():.2f}")
print(f"  - 중앙값: {df_result['진료비(천원)'].median():.2f}")
print(f"  - 최소값: {df_result['진료비(천원)'].min():.2f}")
print(f"  - 최대값: {df_result['진료비(천원)'].max():.2f}")

# ----------------------------------------------------------------------
# 6) 데이터 품질 확인
# ----------------------------------------------------------------------
print("=== 데이터 품질 확인 ===")

# 컬럼 정보 확인
print("\n최종 데이터 컬럼:")
for i, col in enumerate(df_result.columns, 1):
    print(f"{i:2d}. {col}")

# 샘플 데이터 확인
print("\n샘플 데이터 (상위 5행):")
print(df_result.head())

print("\n=== 작업 완료 ===")
