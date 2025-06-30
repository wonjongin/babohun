# -*- coding: utf-8 -*-
"""
진료비 분포 분석 및 구간 설정
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# ----------------------------------------------------------------------
# 1) 데이터 로드
# ----------------------------------------------------------------------
print("=== 데이터 로딩 ===")

# 진료비 데이터 로드
df = pd.read_csv("new_merged_data/df_result2_with_심평원_진료비.csv", encoding="utf-8-sig")

print(f"전체 데이터 행 수: {len(df)}")
print(f"컬럼: {list(df.columns)}")

# ----------------------------------------------------------------------
# 2) 진료비 분포 분석
# ----------------------------------------------------------------------
print("\n=== 진료비 분포 분석 ===")

# 진료비가 있는 데이터만 필터링
df_with_cost = df[df['진료비(천원)'].notna()].copy()
print(f"진료비가 있는 데이터: {len(df_with_cost)}개")

# 진료비 통계
cost_stats = df_with_cost['진료비(천원)'].describe()
print("\n진료비 통계:")
print(cost_stats)

# 진료비 분포 시각화
plt.figure(figsize=(15, 10))

# 1. 히스토그램 (원본 스케일)
plt.subplot(2, 3, 1)
plt.hist(df_with_cost['진료비(천원)'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('진료비 분포 (원본 스케일)')
plt.xlabel('진료비(천원)')
plt.ylabel('빈도')
plt.yscale('log')

# 2. 로그 스케일 히스토그램
plt.subplot(2, 3, 2)
log_cost = np.log1p(df_with_cost['진료비(천원)'])  # log(1+x) 사용
plt.hist(log_cost, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
plt.title('진료비 분포 (로그 스케일)')
plt.xlabel('log(진료비+1)')
plt.ylabel('빈도')

# 3. 박스플롯
plt.subplot(2, 3, 3)
plt.boxplot(df_with_cost['진료비(천원)'])
plt.title('진료비 박스플롯')
plt.ylabel('진료비(천원)')

# 4. 구분별 진료비 분포
plt.subplot(2, 3, 4)
df_with_cost.boxplot(column='진료비(천원)', by='구분', ax=plt.gca())
plt.title('구분별 진료비 분포')
plt.suptitle('')

# 5. 진료과별 진료비 분포 (상위 10개)
plt.subplot(2, 3, 5)
top_depts = df_with_cost.groupby('진료과')['진료비(천원)'].mean().sort_values(ascending=False).head(10)
top_depts.plot(kind='bar', color='orange', alpha=0.7)
plt.title('진료과별 평균 진료비 (상위 10개)')
plt.xticks(rotation=45, ha='right')
plt.ylabel('평균 진료비(천원)')

# 6. 연도별 진료비 트렌드
plt.subplot(2, 3, 6)
yearly_cost = df_with_cost.groupby('년도')['진료비(천원)'].mean()
yearly_cost.plot(kind='line', marker='o', color='red', linewidth=2, markersize=6)
plt.title('연도별 평균 진료비 트렌드')
plt.xlabel('연도')
plt.ylabel('평균 진료비(천원)')

plt.tight_layout()
plt.savefig('new_merged_data/진료비_분포_분석.png', dpi=300, bbox_inches='tight')
plt.show()

# ----------------------------------------------------------------------
# 3) 로그 스케일 기반 구간 설정
# ----------------------------------------------------------------------
print("\n=== 로그 스케일 기반 구간 설정 ===")

# 로그 스케일로 변환
df_with_cost['log_cost'] = np.log1p(df_with_cost['진료비(천원)'])

# 구간 설정 (10개 구간)
n_bins = 10
df_with_cost['cost_bin'] = pd.qcut(df_with_cost['log_cost'], q=n_bins, labels=False, duplicates='drop')

# 구간별 통계
bin_stats = df_with_cost.groupby('cost_bin').agg({
    '진료비(천원)': ['count', 'min', 'max', 'mean', 'median']
}).round(2)

print("\n구간별 통계:")
print(bin_stats)

# 구간별 경계값 확인
bin_boundaries = df_with_cost.groupby('cost_bin')['진료비(천원)'].agg(['min', 'max']).round(2)
print("\n구간별 경계값:")
print(bin_boundaries)

# ----------------------------------------------------------------------
# 4) 구간별 분포 시각화
# ----------------------------------------------------------------------
plt.figure(figsize=(12, 8))

# 구간별 분포
plt.subplot(2, 2, 1)
bin_counts = df_with_cost['cost_bin'].value_counts().sort_index()
bin_counts.plot(kind='bar', color='lightblue', alpha=0.7, edgecolor='black')
plt.title('구간별 데이터 분포')
plt.xlabel('구간 번호')
plt.ylabel('데이터 개수')
plt.xticks(rotation=0)

# 구간별 평균 진료비
plt.subplot(2, 2, 2)
bin_means = df_with_cost.groupby('cost_bin')['진료비(천원)'].mean()
bin_means.plot(kind='bar', color='lightgreen', alpha=0.7, edgecolor='black')
plt.title('구간별 평균 진료비')
plt.xlabel('구간 번호')
plt.ylabel('평균 진료비(천원)')
plt.xticks(rotation=0)

# 구간별 박스플롯
plt.subplot(2, 2, 3)
df_with_cost.boxplot(column='진료비(천원)', by='cost_bin', ax=plt.gca())
plt.title('구간별 진료비 분포')
plt.suptitle('')

# 구간별 로그 스케일 분포
plt.subplot(2, 2, 4)
df_with_cost.boxplot(column='log_cost', by='cost_bin', ax=plt.gca())
plt.title('구간별 로그 스케일 진료비 분포')
plt.suptitle('')

plt.tight_layout()
plt.savefig('new_merged_data/구간별_분포.png', dpi=300, bbox_inches='tight')
plt.show()

# ----------------------------------------------------------------------
# 5) 구간별 상세 정보 출력
# ----------------------------------------------------------------------
print("\n=== 구간별 상세 정보 ===")

for bin_num in sorted(df_with_cost['cost_bin'].unique()):
    bin_data = df_with_cost[df_with_cost['cost_bin'] == bin_num]
    print(f"\n구간 {bin_num}:")
    print(f"  데이터 개수: {len(bin_data)}")
    print(f"  진료비 범위: {bin_data['진료비(천원)'].min():.0f} ~ {bin_data['진료비(천원)'].max():.0f} 천원")
    print(f"  평균 진료비: {bin_data['진료비(천원)'].mean():.0f} 천원")
    print(f"  중앙값: {bin_data['진료비(천원)'].median():.0f} 천원")

# ----------------------------------------------------------------------
# 6) 구간별 진료과 분포
# ----------------------------------------------------------------------
print("\n=== 구간별 진료과 분포 ===")

dept_bin_dist = pd.crosstab(df_with_cost['진료과'], df_with_cost['cost_bin'], normalize='index') * 100
print("\n진료과별 구간 분포 (%):")
print(dept_bin_dist.round(1))

# ----------------------------------------------------------------------
# 7) 결과 저장
# ----------------------------------------------------------------------
print("\n=== 결과 저장 ===")

# 구간 정보가 포함된 데이터 저장
output_path = "new_merged_data/df_result2_with_심평원_진료비_구간.csv"
df_with_cost.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"구간 정보가 포함된 데이터가 '{output_path}'에 저장되었습니다!")
print(f"총 {len(df_with_cost)}개 데이터에 {n_bins}개 구간이 설정되었습니다.")

# 구간별 요약 정보 저장
bin_summary = df_with_cost.groupby('cost_bin').agg({
    '진료비(천원)': ['count', 'min', 'max', 'mean', 'median', 'std']
}).round(2)
bin_summary.columns = ['count', 'min', 'max', 'mean', 'median', 'std']
bin_summary.to_csv('new_merged_data/구간별_요약.csv', encoding='utf-8-sig')

print("구간별 요약 정보가 'new_merged_data/구간별_요약.csv'에 저장되었습니다!") 