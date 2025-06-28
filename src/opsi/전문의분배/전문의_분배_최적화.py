import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from pulp import *
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Pretendard'
plt.rcParams['axes.unicode_minus'] = False

print("=== 전문의 분배 최적화 모델 ===")
print("📊 정수계획법을 사용한 전문의 효율적 분배 시스템")
print()

# --------------------------------------------------
# 1) 데이터 로드 및 전처리
# --------------------------------------------------
print("1/6: 데이터 로드 및 전처리 중...")

# 기존 예측 결과 로드
df_pred = pd.read_csv('analysis_data/병원별_진료과별_입원외래_통합_시계열예측결과_개선.csv')
df_info = pd.read_csv('new_merged_data/병원_통합_데이터.csv')

# 최근 연도(2023)만 사용
df_pred = df_pred[df_pred['연도'] == 2023]

# 병원명 컬럼명 통일
df_pred['병원명'] = df_pred['병원'].astype(str).str.strip()
df_pred['진료과'] = df_pred['진료과'].astype(str).str.strip()
df_info['병원명'] = df_info['병원명'].astype(str).str.strip()

print(f"✅ 데이터 로드 완료")
print(f"  - 예측 데이터: {df_pred.shape}")
print(f"  - 병원 정보: {df_info.shape}")
print()

# --------------------------------------------------
# 2) 현재 상황 분석
# --------------------------------------------------
print("2/6: 현재 상황 분석 중...")

def get_doc_col(진료과):
    return f"{진료과}_전문의수"

# 현재 전문의 현황 분석
current_situation = []
total_doctors = 0
total_patients = 0

for idx, row in df_pred.iterrows():
    병원 = row['병원명']
    진료과 = row['진료과']
    예측환자수 = row['XGB예측']  # 가장 정확한 예측값 사용
    
    info_row = df_info[df_info['병원명'] == 병원]
    doc_col = get_doc_col(진료과)
    
    if len(info_row) > 0 and doc_col in info_row.columns:
        현재전문의수 = info_row.iloc[0][doc_col]
        if pd.notnull(현재전문의수):
            current_situation.append({
                '병원명': 병원,
                '진료과': 진료과,
                '현재전문의수': 현재전문의수,
                '예측환자수': 예측환자수,
                '환자당전문의비율': 예측환자수 / (현재전문의수 + 1)  # 0으로 나누기 방지
            })
            total_doctors += 현재전문의수
            total_patients += 예측환자수

current_df = pd.DataFrame(current_situation)

print(f"✅ 현재 상황 분석 완료")
print(f"  - 총 전문의 수: {total_doctors:.0f}명")
print(f"  - 총 예측 환자 수: {total_patients:.0f}명")
print(f"  - 평균 환자당 전문의 비율: {total_patients/total_doctors:.2f}")
print()

# --------------------------------------------------
# 3) 최적화 모델 설정
# --------------------------------------------------
print("3/6: 최적화 모델 설정 중...")

# 목표: 전문의 효율적 분배
# 제약조건:
# 1. 총 전문의 수는 현재와 동일 (예산 제약)
# 2. 각 진료과별 최소 전문의 수 보장
# 3. 환자당 전문의 비율의 범위 최소화 (공정성)

# 최적화 문제 정의
prob = LpProblem("전문의_분배_최적화", LpMinimize)

# 의사결정 변수: 각 병원-진료과별 전문의 수
doctors = {}
for idx, row in current_df.iterrows():
    key = f"{row['병원명']}_{row['진료과']}"
    doctors[key] = LpVariable(f"doctors_{key}", lowBound=1, cat='Integer')  # 최소 1명

# 목적 함수: 환자당 전문의 비율의 범위 최소화
# 최대 비율과 최소 비율의 차이를 최소화
ratios = []
for idx, row in current_df.iterrows():
    key = f"{row['병원명']}_{row['진료과']}"
    ratio = doctors[key] / (row['예측환자수'] + 1)
    ratios.append(ratio)

# 최대 비율과 최소 비율 변수 추가
max_ratio = LpVariable("max_ratio", lowBound=0)
min_ratio = LpVariable("min_ratio", lowBound=0)

# 목적 함수: 최대 비율과 최소 비율의 차이 최소화
prob += max_ratio - min_ratio

# 제약조건: 각 비율이 최대/최소 범위 내에 있도록
for ratio in ratios:
    prob += ratio <= max_ratio
    prob += ratio >= min_ratio

# 제약조건 1: 총 전문의 수는 현재와 동일
prob += lpSum([doctors[key] for key in doctors.keys()]) == total_doctors

# 제약조건 2: 각 진료과별 최소 전문의 수 보장
진료과별_최소전문의 = {}
for 진료과 in current_df['진료과'].unique():
    진료과_현재 = current_df[current_df['진료과'] == 진료과]['현재전문의수'].sum()
    진료과별_최소전문의[진료과] = max(1, int(진료과_현재 * 0.8))  # 현재의 80% 이상 보장
    
    진료과_doctors = [doctors[key] for key in doctors.keys() if key.split('_')[1] == 진료과]
    prob += lpSum(진료과_doctors) >= 진료과별_최소전문의[진료과]

# 제약조건 3: 각 병원별 최소 전문의 수 보장
병원별_최소전문의 = {}
for 병원 in current_df['병원명'].unique():
    병원_현재 = current_df[current_df['병원명'] == 병원]['현재전문의수'].sum()
    병원별_최소전문의[병원] = max(1, int(병원_현재 * 0.7))  # 현재의 70% 이상 보장
    
    병원_doctors = [doctors[key] for key in doctors.keys() if key.split('_')[0] == 병원]
    prob += lpSum(병원_doctors) >= 병원별_최소전문의[병원]

print(f"✅ 최적화 모델 설정 완료")
print(f"  - 의사결정 변수: {len(doctors)}개")
print(f"  - 제약조건: 총 전문의 수, 진료과별 최소, 병원별 최소")
print()

# --------------------------------------------------
# 4) 최적화 실행
# --------------------------------------------------
print("4/6: 최적화 실행 중...")

# 최적화 실행
prob.solve(PULP_CBC_CMD(msg=False))

print(f"✅ 최적화 완료")
print(f"  - 최적화 상태: {LpStatus[prob.status]}")
print(f"  - 목적 함수 값: {value(prob.objective):.4f}")
print()

# --------------------------------------------------
# 5) 결과 분석
# --------------------------------------------------
print("5/6: 결과 분석 중...")

# 최적화 결과 추출
optimization_results = []
for idx, row in current_df.iterrows():
    key = f"{row['병원명']}_{row['진료과']}"
    최적전문의수 = value(doctors[key])
    
    # 0으로 나누기 방지
    현재전문의수 = row['현재전문의수']
    if 현재전문의수 == 0:
        변화율 = 100.0  # 0에서 증가하는 경우 100% 증가로 설정
    else:
        변화율 = (최적전문의수 - 현재전문의수) / 현재전문의수 * 100
    
    optimization_results.append({
        '병원명': row['병원명'],
        '진료과': row['진료과'],
        '현재전문의수': 현재전문의수,
        '최적전문의수': 최적전문의수,
        '변화량': 최적전문의수 - 현재전문의수,
        '변화율': 변화율,
        '예측환자수': row['예측환자수'],
        '현재_환자당전문의비율': row['예측환자수'] / (현재전문의수 + 1),
        '최적_환자당전문의비율': row['예측환자수'] / (최적전문의수 + 1)
    })

results_df = pd.DataFrame(optimization_results)

# 성능 지표 계산
현재_비율_표준편차 = results_df['현재_환자당전문의비율'].std()
최적_비율_표준편차 = results_df['최적_환자당전문의비율'].std()
개선도 = (현재_비율_표준편차 - 최적_비율_표준편차) / 현재_비율_표준편차 * 100

print(f"✅ 결과 분석 완료")
print(f"  - 현재 비율 표준편차: {현재_비율_표준편차:.4f}")
print(f"  - 최적 비율 표준편차: {최적_비율_표준편차:.4f}")
print(f"  - 개선도: {개선도:.1f}%")
print()

# --------------------------------------------------
# 6) 결과 저장 및 시각화
# --------------------------------------------------
print("6/6: 결과 저장 및 시각화 중...")

# 결과 저장 디렉토리 생성
results_dir = "optimization_results"
os.makedirs(results_dir, exist_ok=True)

# 1) 상세 결과 저장
results_df.to_csv(f"{results_dir}/전문의_분배_최적화_결과.csv", index=False, encoding='utf-8-sig')

# 2) 요약 통계 저장
summary_stats = {
    "timestamp": datetime.now().isoformat(),
    "total_doctors": int(total_doctors),
    "total_patients": int(total_patients),
    "optimization_status": LpStatus[prob.status],
    "objective_value": float(value(prob.objective)),
    "current_ratio_std": float(현재_비율_표준편차),
    "optimal_ratio_std": float(최적_비율_표준편차),
    "improvement_percentage": float(개선도),
    "total_hospitals": int(len(results_df['병원명'].unique())),
    "total_departments": int(len(results_df['진료과'].unique())),
    "doctors_increased": int(len(results_df[results_df['변화량'] > 0])),
    "doctors_decreased": int(len(results_df[results_df['변화량'] < 0])),
    "doctors_unchanged": int(len(results_df[results_df['변화량'] == 0]))
}

with open(f"{results_dir}/최적화_요약.json", 'w', encoding='utf-8') as f:
    json.dump(summary_stats, f, ensure_ascii=False, indent=2)

# 3) 시각화
plt.figure(figsize=(15, 10))

# 서브플롯 1: 현재 vs 최적 전문의 수 비교
plt.subplot(2, 2, 1)
plt.scatter(results_df['현재전문의수'], results_df['최적전문의수'], alpha=0.6)
plt.plot([0, results_df['현재전문의수'].max()], [0, results_df['현재전문의수'].max()], 'r--', alpha=0.5)
plt.xlabel('현재 전문의 수')
plt.ylabel('최적 전문의 수')
plt.title('현재 vs 최적 전문의 수 비교')
plt.grid(True, alpha=0.3)

# 서브플롯 2: 변화량 분포
plt.subplot(2, 2, 2)
plt.hist(results_df['변화량'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('전문의 수 변화량')
plt.ylabel('빈도')
plt.title('전문의 수 변화량 분포')
plt.grid(True, alpha=0.3)

# 서브플롯 3: 환자당 전문의 비율 개선
plt.subplot(2, 2, 3)
plt.scatter(results_df['현재_환자당전문의비율'], results_df['최적_환자당전문의비율'], alpha=0.6)
plt.plot([0, results_df['현재_환자당전문의비율'].max()], [0, results_df['현재_환자당전문의비율'].max()], 'r--', alpha=0.5)
plt.xlabel('현재 환자당 전문의 비율')
plt.ylabel('최적 환자당 전문의 비율')
plt.title('환자당 전문의 비율 개선')
plt.grid(True, alpha=0.3)

# 서브플롯 4: 진료과별 평균 변화량
plt.subplot(2, 2, 4)
dept_changes = results_df.groupby('진료과')['변화량'].mean().sort_values()
plt.barh(dept_changes.index, dept_changes.values, alpha=0.7, color='lightcoral')
plt.xlabel('평균 변화량')
plt.title('진료과별 평균 전문의 수 변화량')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{results_dir}/전문의_분배_최적화_시각화.png", dpi=300, bbox_inches='tight')
plt.show()

# 4) 상세 분석 리포트 생성
print("📊 상세 분석 리포트 생성 중...")

# 상위 10개 개선 사례
top_improvements = results_df.nlargest(10, '변화량')
print("\n=== 상위 10개 전문의 증가 사례 ===")
print(top_improvements[['병원명', '진료과', '현재전문의수', '최적전문의수', '변화량', '변화율']].to_string(index=False))

# 하위 10개 감소 사례
top_decreases = results_df.nsmallest(10, '변화량')
print("\n=== 상위 10개 전문의 감소 사례 ===")
print(top_decreases[['병원명', '진료과', '현재전문의수', '최적전문의수', '변화량', '변화율']].to_string(index=False))

# 진료과별 요약
dept_summary = results_df.groupby('진료과').agg({
    '현재전문의수': 'sum',
    '최적전문의수': 'sum',
    '변화량': 'sum',
    '예측환자수': 'sum'
}).round(2)

dept_summary['현재_환자당전문의비율'] = dept_summary['예측환자수'] / (dept_summary['현재전문의수'] + 1)
dept_summary['최적_환자당전문의비율'] = dept_summary['예측환자수'] / (dept_summary['최적전문의수'] + 1)
dept_summary['개선도'] = (dept_summary['현재_환자당전문의비율'] - dept_summary['최적_환자당전문의비율']) / dept_summary['현재_환자당전문의비율'] * 100

print("\n=== 진료과별 요약 ===")
print(dept_summary.to_string())

# 진료과별 요약 저장
dept_summary.to_csv(f"{results_dir}/진료과별_요약.csv", encoding='utf-8-sig')

print(f"\n✅ 모든 결과가 {results_dir}/ 디렉토리에 저장되었습니다!")
print("="*60)
print("🎯 전문의 분배 최적화 완료!")
print("="*60) 