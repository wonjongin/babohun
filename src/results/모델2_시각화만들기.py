# -*- coding: utf-8 -*-
"""
모델2 시각화: ARIMA 예측 결과와 이전 추이 시각화
2013-2023년 실제 데이터 (실선) + 2024-2026년 ARIMA 예측 (점선)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import font_manager, rc
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 - Pretendard
plt.rcParams['font.family'] = 'Pretendard'
plt.rcParams['axes.unicode_minus'] = False

# ----------------------------------------------------------------------
# 1) 데이터 로드
# ----------------------------------------------------------------------
print("=== 데이터 로딩 ===")

# 이전 데이터 (2013-2023년)
historical_data = pd.read_csv('new_merged_data/연도별진료과별_입원외래_통합환자수.csv', encoding='utf-8-sig')

# 예측 데이터 (2024-2026년)
prediction_data = pd.read_csv('analysis_data/병원별_진료과별_미래3년_예측결과.csv', encoding='utf-8-sig')

print(f"이전 데이터: {len(historical_data)}행")
print(f"예측 데이터: {len(prediction_data)}행")

# ----------------------------------------------------------------------
# 2) 데이터 전처리
# ----------------------------------------------------------------------
print("=== 데이터 전처리 ===")

# 이전 데이터에서 2013-2023년만 필터링
historical_filtered = historical_data[historical_data['연도'].isin(range(2013, 2024))].copy()
historical_filtered = historical_filtered[['연도', '병원', '진료과', '전체환자수_합계']]
historical_filtered.columns = ['연도', '병원', '진료과', '환자수']

# 예측 데이터에서 ARIMA 예측만 선택
prediction_filtered = prediction_data[['병원', '진료과', '예측연도', 'ARIMA예측']].copy()
prediction_filtered.columns = ['병원', '진료과', '연도', '환자수']

# 데이터 통합
combined_data = pd.concat([historical_filtered, prediction_filtered], ignore_index=True)

# 데이터 타입 변환
combined_data['연도'] = combined_data['연도'].astype(int)
combined_data['환자수'] = pd.to_numeric(combined_data['환자수'], errors='coerce')

print(f"통합 데이터: {len(combined_data)}행")
print(f"병원 수: {combined_data['병원'].nunique()}")
print(f"진료과 수: {combined_data['진료과'].nunique()}")

# ----------------------------------------------------------------------
# 3) 주요 진료과 선정
# ----------------------------------------------------------------------
print("=== 주요 진료과 선정 ===")

# 전체 환자수 기준으로 상위 진료과 선정
top_departments = historical_filtered.groupby('진료과')['환자수'].sum().sort_values(ascending=False).head(15)
print("상위 15개 진료과:")
print(top_departments)

# 주요 진료과 리스트
major_departments = [
    '내과', '정형외과', '재활의학과', '신경과', '신경외과', 
    '안과', '비뇨기과', '치과', '피부과', '이비인후과',
    '가정의학과', '외과', '정신건강의학과', '소아청소년과', '산부인과'
]

# ----------------------------------------------------------------------
# 4) 종합 서브플롯 시각화 생성
# ----------------------------------------------------------------------
print("=== 종합 서브플롯 시각화 생성 ===")

hospitals = ['광주', '대구', '대전', '부산', '중앙', '인천']

fig, axes = plt.subplots(3, 5, figsize=(28, 14))
fig.suptitle('보훈병원 진료과별 환자수 추이 및 ARIMA 예측 (2013-2026)', fontsize=22, fontweight='bold', y=0.98)

for idx, department in enumerate(major_departments):
    row = idx // 5
    col = idx % 5
    ax = axes[row, col]
    
    # 해당 진료과의 모든 병원 데이터
    dept_data = combined_data[combined_data['진료과'] == department].copy()
    if len(dept_data) == 0:
        continue
    
    # 병원별로 다른 색상 사용
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8B5A3C', '#4A90E2']
    
    for i, hospital in enumerate(hospitals):
        hospital_data = dept_data[dept_data['병원'] == hospital].copy()
        if len(hospital_data) == 0:
            continue
        hospital_data = hospital_data.sort_values('연도')
        actual = hospital_data[hospital_data['연도'] <= 2023]
        predicted = hospital_data[hospital_data['연도'] >= 2024]
        if len(actual) > 0:
            ax.plot(actual['연도'], actual['환자수'], marker='o', linewidth=2, markersize=3,
                    color=colors[i], label=f'{hospital} (실제)', alpha=0.8)
        if len(predicted) > 0:
            ax.plot(predicted['연도'], predicted['환자수'], marker='s', linewidth=2, markersize=3, linestyle='--',
                    color=colors[i], label=f'{hospital} (예측)', alpha=0.8)
        if len(actual) > 0 and len(predicted) > 0:
            last_actual = actual.iloc[-1]
            first_predicted = predicted.iloc[0]
            ax.plot([last_actual['연도'], first_predicted['연도']],
                    [last_actual['환자수'], first_predicted['환자수']],
                    '--', color=colors[i], alpha=0.5, linewidth=1)
    ax.set_title(department, fontsize=16, fontweight='bold')
    ax.set_xlabel('연도', fontsize=12)
    ax.set_ylabel('환자수', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)  # 범례를 왼쪽 위로
    years = list(range(2013, 2027, 2))
    ax.set_xticks(years)
    ax.set_xticklabels([str(year) for year in years], rotation=45)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('analysis_data/종합_진료과별_환자수_추이_예측.png', dpi=300, bbox_inches='tight')

# ----------------------------------------------------------------------
# 5) 통계 요약 생성
# ----------------------------------------------------------------------
print("=== 통계 요약 생성 ===")

summary_stats = []
for hospital in hospitals:
    for department in major_departments:
        actual_2023 = historical_filtered[
            (historical_filtered['병원'] == hospital) & 
            (historical_filtered['진료과'] == department) & 
            (historical_filtered['연도'] == 2023)
        ]['환자수'].iloc[0] if len(historical_filtered[
            (historical_filtered['병원'] == hospital) & 
            (historical_filtered['진료과'] == department) & 
            (historical_filtered['연도'] == 2023)
        ]) > 0 else None
        predicted_2024 = prediction_filtered[
            (prediction_filtered['병원'] == hospital) & 
            (prediction_filtered['진료과'] == department) & 
            (prediction_filtered['연도'] == 2024)
        ]['환자수'].iloc[0] if len(prediction_filtered[
            (prediction_filtered['병원'] == hospital) & 
            (prediction_filtered['진료과'] == department) & 
            (prediction_filtered['연도'] == 2024)
        ]) > 0 else None
        if actual_2023 is not None and predicted_2024 is not None:
            change_rate = ((predicted_2024 - actual_2023) / actual_2023) * 100
            summary_stats.append({
                '병원': hospital,
                '진료과': department,
                '2023년_실제': actual_2023,
                '2024년_예측': predicted_2024,
                '변화율(%)': change_rate
            })
summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv('analysis_data/ARIMA_예측_요약통계.csv', index=False, encoding='utf-8-sig')

print("✅ 시각화 완료!")
print(f"- 종합 서브플롯 그래프: 1개")
print(f"- 통계 요약: {len(summary_df)}개 진료과")
print(f"- 파일 저장 위치: analysis_data/ 폴더")
