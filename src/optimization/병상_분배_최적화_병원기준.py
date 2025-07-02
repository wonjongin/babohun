import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import pulp
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpStatus, value, PULP_CBC_CMD
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Pretendard'
plt.rcParams['axes.unicode_minus'] = False

print("=== 병상 분배 최적화 모델 (병원 단위) ===")
print("📊 정수계획법을 사용한 병상 효율적 분배 시스템 (진료과 무시, 병원별 총 병상만)")
print()

# --------------------------------------------------
# 1) 데이터 로드 및 전처리
# --------------------------------------------------
print("1/5: 데이터 로드 및 전처리 중...")

# 병원 통합 데이터 로드 (병상 현황)
df_hospital = pd.read_csv('new_merged_data/병원_통합_데이터_호스피스 삭제.csv')

# 입원 예측 데이터 로드
df_pred = pd.read_csv('analysis_data/병원별_진료과별_입원_미래3년_예측결과.csv')

# 최근 연도(2024)만 사용
df_pred = df_pred[df_pred['예측연도'] == 2024]

# 병원명 컬럼명 통일
df_pred['병원명'] = df_pred['병원'].astype(str).str.strip()
df_hospital['병원명'] = df_hospital['병원명'].astype(str).str.strip()

# 병상 관련 컬럼 추출 (전문의수로 끝나지 않는 컬럼들)
bed_columns = [col for col in df_hospital.columns if not col.endswith('_전문의수') and col != '병원명']
df_hospital['총병상수'] = df_hospital[bed_columns].sum(axis=1)

# 병원별 예측 환자 수 집계
hospital_patients = df_pred.groupby('병원명')['XGB예측'].sum().reset_index()
hospital_patients.columns = ['병원명', '총예측환자수']

# 병원별 현재 상황 분석
data = []
total_beds = 0
total_patients = 0
for idx, row in hospital_patients.iterrows():
    병원 = row['병원명']
    예측환자수 = row['총예측환자수']
    hosp_row = df_hospital[df_hospital['병원명'] == 병원]
    if len(hosp_row) > 0:
        현재병상수 = hosp_row['총병상수'].values[0]
        if pd.notnull(현재병상수) and 현재병상수 > 0:
            data.append({
                '병원명': 병원,
                '현재병상수': 현재병상수,
                '예측환자수': 예측환자수
            })
            total_beds += 현재병상수
            total_patients += 예측환자수
current_df = pd.DataFrame(data)

print(f"✅ 데이터 로드 및 집계 완료: 병원 수 {len(current_df)}개")
print(f"  - 총 병상 수: {total_beds:.0f}개")
print(f"  - 총 예측 환자 수: {total_patients:.0f}명")
print()

# --------------------------------------------------
# 2) 최적화 모델 설정 (병원 단위)
# --------------------------------------------------
print("2/5: 최적화 모델 설정 중...")

prob = LpProblem("병원별_병상_분배_최적화", LpMinimize)

# 의사결정 변수: 각 병원별 병상 수
beds = {}
for idx, row in current_df.iterrows():
    병원명 = row['병원명']
    beds[병원명] = LpVariable(f"beds_{병원명}", lowBound=1, cat='Integer')

# 목적 함수: 병상 수 변화량의 절댓값 합 최소화
total_change = 0
for idx, row in current_df.iterrows():
    병원명 = row['병원명']
    current_beds = row['현재병상수']
    change_positive = LpVariable(f"change_pos_{병원명}", lowBound=0)
    change_negative = LpVariable(f"change_neg_{병원명}", lowBound=0)
    prob += beds[병원명] - current_beds == change_positive - change_negative
    total_change += change_positive + change_negative
prob += total_change

# 제약조건 1: 총 병상 수는 현재와 동일
prob += lpSum([beds[병원명] for 병원명 in beds.keys()]) == total_beds

# 제약조건 2: 각 병원별 최소 병상 수 보장 (현재의 70% 이상)
for idx, row in current_df.iterrows():
    병원명 = row['병원명']
    현재병상수 = row['현재병상수']
    최소병상수 = max(1, int(현재병상수 * 0.7))
    prob += beds[병원명] >= 최소병상수

print(f"✅ 최적화 모델 설정 완료 (병원별 변수 {len(beds)}개)")
print()

# --------------------------------------------------
# 3) 최적화 실행
# --------------------------------------------------
print("3/5: 최적화 실행 중...")
try:
    prob.solve(PULP_CBC_CMD(msg=False))
    print("✅ CBC 솔버 최적화 완료!")
except Exception as e:
    print(f"⚠️  CBC 솔버 오류: {e}")
    prob.solve()
print(f"  - 최적화 상태: {LpStatus[prob.status]}")
print(f"  - 목적 함수 값: {value(prob.objective):.4f}")
print()

# --------------------------------------------------
# 4) 결과 분석 및 저장
# --------------------------------------------------
print("4/5: 결과 분석 및 저장 중...")

results = []
for idx, row in current_df.iterrows():
    병원명 = row['병원명']
    최적병상수 = value(beds[병원명])
    현재병상수 = row['현재병상수']
    예측환자수 = row['예측환자수']
    변화량 = 최적병상수 - 현재병상수
    변화율 = (변화량 / 현재병상수 * 100) if 현재병상수 != 0 else 0
    일평균환자수 = 예측환자수 / 365
    현재_가동률 = (일평균환자수 / (현재병상수 + 1)) * 100
    최적_가동률 = (일평균환자수 / (최적병상수 + 1)) * 100
    results.append({
        '병원명': 병원명,
        '현재병상수': 현재병상수,
        '최적병상수': 최적병상수,
        '변화량': 변화량,
        '변화율': 변화율,
        '예측환자수': 예측환자수,
        '현재_병상가동률': 현재_가동률,
        '최적_병상가동률': 최적_가동률
    })
results_df = pd.DataFrame(results)

results_dir = "optimization_results_병상_분배_최적화_병원기준"
os.makedirs(results_dir, exist_ok=True)
results_df.to_csv(f"{results_dir}/병상_분배_최적화_결과.csv", index=False, encoding='utf-8-sig')

print(f"✅ 결과 저장 완료: {results_dir}/병상_분배_최적화_결과.csv")
print()

# --------------------------------------------------
# 5) 시각화 및 리포트
# --------------------------------------------------
print("5/5: 시각화 및 리포트 생성 중...")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(results_df['현재병상수'], results_df['최적병상수'], alpha=0.7)
max_beds = max(results_df['현재병상수'].max(), results_df['최적병상수'].max())
plt.plot([0, max_beds], [0, max_beds], 'r--', alpha=0.5)
plt.xlabel('현재 병상 수')
plt.ylabel('최적 병상 수')
plt.title('현재 vs 최적 병상 수 (병원별)')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.barh(results_df['병원명'], results_df['변화량'], alpha=0.7, color='lightcoral')
plt.xlabel('병상 수 변화량')
plt.title('병원별 병상 수 변화량')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{results_dir}/병상_분배_최적화_시각화.png", dpi=300, bbox_inches='tight')
plt.show()

print("\n=== 병원별 최적화 결과 요약 ===")
print(results_df[['병원명', '현재병상수', '최적병상수', '변화량', '변화율', '현재_병상가동률', '최적_병상가동률']].round(2).to_string(index=False))

print(f"\n✅ 모든 결과가 {results_dir}/ 디렉토리에 저장되었습니다!")
print("="*60)
print("🎯 병상 분배 최적화(병원 단위) 완료!")
print("="*60)
