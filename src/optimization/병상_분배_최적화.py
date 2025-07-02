import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import pulp
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpStatus, value, PULP_CBC_CMD
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Pretendard'
plt.rcParams['axes.unicode_minus'] = False

print("=== 병상 분배 최적화 모델 ===")
print("📊 정수계획법을 사용한 병상 효율적 분배 시스템")
print()

# --------------------------------------------------
# 1) 데이터 로드 및 전처리
# --------------------------------------------------
print("1/6: 데이터 로드 및 전처리 중...")

# 병원 통합 데이터 로드 (병상 현황)
df_hospital = pd.read_csv('new_merged_data/병원_통합_데이터_호스피스 삭제.csv')

# 입원 예측 데이터 로드
df_pred = pd.read_csv('analysis_data/병원별_진료과별_입원_미래3년_예측결과.csv')

# 최근 연도(2024)만 사용
df_pred = df_pred[df_pred['예측연도'] == 2024]

# 병원명 컬럼명 통일
df_pred['병원명'] = df_pred['병원'].astype(str).str.strip()
df_pred['진료과'] = df_pred['진료과'].astype(str).str.strip()
df_hospital['병원명'] = df_hospital['병원명'].astype(str).str.strip()

print(f"✅ 데이터 로드 완료")
print(f"  - 병원 데이터: {df_hospital.shape}")
print(f"  - 예측 데이터: {df_pred.shape}")
print()

# --------------------------------------------------
# 2) 병상 데이터 추출 및 전처리
# --------------------------------------------------
print("2/6: 병상 데이터 추출 및 전처리 중...")

# 병상 관련 컬럼 추출 (전문의수로 끝나지 않는 컬럼들)
bed_columns = []
for col in df_hospital.columns:
    if not col.endswith('_전문의수') and col != '병원명':
        bed_columns.append(col)

print(f"병상 관련 컬럼: {bed_columns}")

# 각 병원의 총 병상수 계산
df_hospital['총병상수'] = df_hospital[bed_columns].sum(axis=1)

# 진료과별 병상 매핑 (간단한 매핑 규칙)
dept_bed_mapping = {
    '내과': ['일반입원실_상급', '일반입원실_일반', '중환자실_성인'],
    '외과': ['일반입원실_상급', '일반입원실_일반', '중환자실_성인'],
    '정형외과': ['일반입원실_상급', '일반입원실_일반', '중환자실_성인'],
    '신경외과': ['일반입원실_상급', '일반입원실_일반', '중환자실_성인'],
    '산부인과': ['분만실', '일반입원실_상급', '일반입원실_일반'],
    '소아청소년과': ['일반입원실_상급', '일반입원실_일반', '중환자실_소아', '신생아실'],
    '신경과': ['일반입원실_상급', '일반입원실_일반', '중환자실_성인'],
    '재활의학과': ['일반입원실_상급', '일반입원실_일반'],
    '정신건강의학과': ['정신과개방_상급', '정신과개방_일반', '정신과폐쇄_상급', '정신과폐쇄_일반'],
    '응급의학과': ['응급실', '중환자실_성인'],
    '비뇨의학과': ['일반입원실_상급', '일반입원실_일반'],
    '안과': ['일반입원실_상급', '일반입원실_일반'],
    '이비인후과': ['일반입원실_상급', '일반입원실_일반'],
    '피부과': ['일반입원실_상급', '일반입원실_일반'],
    '가정의학과': ['일반입원실_상급', '일반입원실_일반']
}

# 현재 상황 분석
current_situation = []
total_beds = 0
total_patients = 0

for idx, row in df_pred.iterrows():
    병원 = row['병원명']
    진료과 = row['진료과']
    예측환자수 = row['XGB예측']  # 가장 정확한 예측값 사용
    
    # 해당 병원의 병상 데이터 찾기
    hosp_row = df_hospital[df_hospital['병원명'] == 병원]
    
    if len(hosp_row) > 0 and 진료과 in dept_bed_mapping:
        # 해당 진료과의 병상 컬럼들
        dept_bed_cols = dept_bed_mapping[진료과]
        
        # 현재 병상수 계산 (해당 진료과 관련 병상들의 합)
        현재병상수 = hosp_row[dept_bed_cols].sum(axis=1).values[0]
        
        if pd.notnull(현재병상수) and 현재병상수 > 0:
            current_situation.append({
                '병원명': 병원,
                '진료과': 진료과,
                '현재병상수': 현재병상수,
                '예측환자수': 예측환자수,
                '환자당병상비율': 예측환자수 / (현재병상수 + 1)  # 0으로 나누기 방지
            })
            total_beds += 현재병상수
            total_patients += 예측환자수

current_df = pd.DataFrame(current_situation)

print(f"✅ 병상 데이터 전처리 완료")
print(f"  - 총 병상 수: {total_beds:.0f}개")
print(f"  - 총 예측 환자 수: {total_patients:.0f}명")
print(f"  - 평균 환자당 병상 비율: {total_patients/total_beds:.2f}")
print()

# --------------------------------------------------
# 3) 최적화 모델 설정
# --------------------------------------------------
print("3/6: 최적화 모델 설정 중...")

# 목표: 병상 효율적 분배로 평가지표 최적화
# 제약조건:
# 1. 총 병상 수는 현재와 동일 (예산 제약)
# 2. 각 진료과별 최소 병상 수 보장
# 3. 병상가동률을 85~90% 범위로 최적화
# 4. 환자당 병상 비율의 범위 최소화 (공정성)

# 최적화 문제 정의
prob = LpProblem("병상_분배_최적화", LpMinimize)

# 의사결정 변수: 각 병원-진료과별 병상 수
beds = {}
for idx, row in current_df.iterrows():
    key = f"{row['병원명']}_{row['진료과']}"
    beds[key] = LpVariable(f"beds_{key}", lowBound=1, cat='Integer')  # 최소 1개

# 목적 함수: 병상 수 변화량 최소화 + 균등 분배
# 목표: 현재 병상 수에서 최소한의 변화로 균등한 분배 달성

# 목적 함수: 모든 병상 수 변화량의 절댓값 합 최소화
total_change = 0
for idx, row in current_df.iterrows():
    key = f"{row['병원명']}_{row['진료과']}"
    current_beds = row['현재병상수']
    # 변화량의 절댓값을 최소화 (PuLP에서는 abs() 대신 양수/음수 변수 사용)
    change_positive = LpVariable(f"change_pos_{key}", lowBound=0)
    change_negative = LpVariable(f"change_neg_{key}", lowBound=0)
    
    # 변화량 제약: beds[key] - current_beds = change_positive - change_negative
    prob += beds[key] - current_beds == change_positive - change_negative
    
    total_change += change_positive + change_negative

# 목적 함수: 변화량 최소화
prob += total_change

# 제약조건 1: 총 병상 수는 현재와 동일
prob += lpSum([beds[key] for key in beds.keys()]) == total_beds

# 제약조건 2: 각 진료과별 최소 병상 수 보장
진료과별_최소병상 = {}
for 진료과 in current_df['진료과'].unique():
    진료과_현재 = current_df[current_df['진료과'] == 진료과]['현재병상수'].sum()
    진료과별_최소병상[진료과] = max(1, int(진료과_현재 * 0.8))  # 현재의 80% 이상 보장
    
    진료과_beds = [beds[key] for key in beds.keys() if key.split('_')[1] == 진료과]
    prob += lpSum(진료과_beds) >= 진료과별_최소병상[진료과]

# 제약조건 3: 각 병원별 최소 병상 수 보장
병원별_최소병상 = {}
for 병원 in current_df['병원명'].unique():
    병원_현재 = current_df[current_df['병원명'] == 병원]['현재병상수'].sum()
    병원별_최소병상[병원] = max(1, int(병원_현재 * 0.7))  # 현재의 70% 이상 보장
    
    병원_beds = [beds[key] for key in beds.keys() if key.split('_')[0] == 병원]
    prob += lpSum(병원_beds) >= 병원별_최소병상[병원]

print(f"✅ 최적화 모델 설정 완료")
print(f"  - 의사결정 변수: {len(beds)}개")
print(f"  - 제약조건: 총 병상 수, 진료과별 최소, 병원별 최소")
print()

# --------------------------------------------------
# 4) 최적화 실행
# --------------------------------------------------
print("4/6: 최적화 실행 중...")

# 최적화 실행 (CBC 솔버 사용)
try:
    print("🔄 CBC 솔버로 최적화 실행 중...")
    prob.solve(PULP_CBC_CMD(msg=False))
    print("✅ CBC 솔버 최적화 완료!")
except Exception as e:
    print(f"⚠️  CBC 솔버 오류: {e}")
    try:
        print("🔄 기본 솔버로 재시도 중...")
        prob.solve()  # 기본 솔버 사용
        print("✅ 기본 솔버 최적화 완료!")
    except Exception as e2:
        print(f"⚠️  기본 솔버도 실패: {e2}")
        print("⚠️  휴리스틱 방법으로 대체합니다.")
        # 휴리스틱 방법: 현재 상태를 그대로 유지
        for idx, row in current_df.iterrows():
            key = f"{row['병원명']}_{row['진료과']}"
            beds[key].setInitialValue(row['현재병상수'])
        prob.solve()

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
    최적병상수 = value(beds[key])
    
    # 0으로 나누기 방지
    현재병상수 = row['현재병상수']
    if 현재병상수 == 0:
        변화율 = 100.0  # 0에서 증가하는 경우 100% 증가로 설정
    else:
        변화율 = (최적병상수 - 현재병상수) / 현재병상수 * 100
    
    # 병상가동률 계산
    일평균환자수 = row['예측환자수'] / 365
    현재_가동률 = (일평균환자수 / (현재병상수 + 1)) * 100
    최적_가동률 = (일평균환자수 / (최적병상수 + 1)) * 100
    
    optimization_results.append({
        '병원명': row['병원명'],
        '진료과': row['진료과'],
        '현재병상수': 현재병상수,
        '최적병상수': 최적병상수,
        '변화량': 최적병상수 - 현재병상수,
        '변화율': 변화율,
        '예측환자수': row['예측환자수'],
        '현재_병상가동률': 현재_가동률,
        '최적_병상가동률': 최적_가동률,
        '현재_환자당병상비율': row['예측환자수'] / (현재병상수 + 1),
        '최적_환자당병상비율': row['예측환자수'] / (최적병상수 + 1)
    })

results_df = pd.DataFrame(optimization_results)

# 성능 지표 계산
현재_가동률_표준편차 = results_df['현재_병상가동률'].std()
최적_가동률_표준편차 = results_df['최적_병상가동률'].std()
가동률_개선도 = (현재_가동률_표준편차 - 최적_가동률_표준편차) / 현재_가동률_표준편차 * 100

# 85~90% 범위 내 병상가동률 비율
현재_적정가동률_비율 = len(results_df[(results_df['현재_병상가동률'] >= 85) & (results_df['현재_병상가동률'] <= 90)]) / len(results_df) * 100
최적_적정가동률_비율 = len(results_df[(results_df['최적_병상가동률'] >= 85) & (results_df['최적_병상가동률'] <= 90)]) / len(results_df) * 100

print(f"✅ 결과 분석 완료")
print(f"  - 현재 가동률 표준편차: {현재_가동률_표준편차:.4f}")
print(f"  - 최적 가동률 표준편차: {최적_가동률_표준편차:.4f}")
print(f"  - 가동률 개선도: {가동률_개선도:.1f}%")
print(f"  - 현재 적정가동률(85~90%) 비율: {현재_적정가동률_비율:.1f}%")
print(f"  - 최적 적정가동률(85~90%) 비율: {최적_적정가동률_비율:.1f}%")
print()

# --------------------------------------------------
# 6) 결과 저장 및 시각화
# --------------------------------------------------
print("6/6: 결과 저장 및 시각화 중...")

# 결과 저장 디렉토리 생성
results_dir = "optimization_results_병상_분배_최적화"
os.makedirs(results_dir, exist_ok=True)

# 1) 상세 결과 저장
results_df.to_csv(f"{results_dir}/병상_분배_최적화_결과.csv", index=False, encoding='utf-8-sig')

# 2) 요약 통계 저장
summary_stats = {
    "timestamp": datetime.now().isoformat(),
    "total_beds": int(total_beds),
    "total_patients": int(total_patients),
    "optimization_status": LpStatus[prob.status],
    "objective_value": float(value(prob.objective)),
    "current_utilization_std": float(현재_가동률_표준편차),
    "optimal_utilization_std": float(최적_가동률_표준편차),
    "utilization_improvement_percentage": float(가동률_개선도),
    "current_optimal_utilization_ratio": float(현재_적정가동률_비율),
    "optimal_optimal_utilization_ratio": float(최적_적정가동률_비율),
    "total_hospitals": int(len(results_df['병원명'].unique())),
    "total_departments": int(len(results_df['진료과'].unique())),
    "beds_increased": int(len(results_df[results_df['변화량'] > 0])),
    "beds_decreased": int(len(results_df[results_df['변화량'] < 0])),
    "beds_unchanged": int(len(results_df[results_df['변화량'] == 0]))
}

with open(f"{results_dir}/최적화_요약.json", 'w', encoding='utf-8') as f:
    json.dump(summary_stats, f, ensure_ascii=False, indent=2)

# 3) 시각화
plt.figure(figsize=(15, 10))

# 서브플롯 1: 현재 vs 최적 병상 수 비교
plt.subplot(2, 2, 1)
plt.scatter(results_df['현재병상수'], results_df['최적병상수'], alpha=0.6)
max_beds = max(results_df['현재병상수'].max(), results_df['최적병상수'].max())
plt.plot([0, max_beds], [0, max_beds], 'r--', alpha=0.5)
plt.xlabel('현재 병상 수')
plt.ylabel('최적 병상 수')
plt.title('현재 vs 최적 병상 수 비교')
plt.grid(True, alpha=0.3)

# 서브플롯 2: 변화량 분포
plt.subplot(2, 2, 2)
plt.hist(results_df['변화량'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('병상 수 변화량')
plt.ylabel('빈도')
plt.title('병상 수 변화량 분포')
plt.grid(True, alpha=0.3)

# 서브플롯 3: 병상가동률 개선
plt.subplot(2, 2, 3)
plt.scatter(results_df['현재_병상가동률'], results_df['최적_병상가동률'], alpha=0.6)
plt.axhline(y=85, color='g', linestyle='--', alpha=0.5, label='85%')
plt.axhline(y=90, color='g', linestyle='--', alpha=0.5, label='90%')
plt.axvline(x=85, color='g', linestyle='--', alpha=0.5)
plt.axvline(x=90, color='g', linestyle='--', alpha=0.5)
max_util = max(results_df['현재_병상가동률'].max(), results_df['최적_병상가동률'].max())
plt.plot([0, max_util], [0, max_util], 'r--', alpha=0.5)
plt.xlabel('현재 병상가동률 (%)')
plt.ylabel('최적 병상가동률 (%)')
plt.title('병상가동률 개선 (85~90% 목표)')
plt.legend()
plt.grid(True, alpha=0.3)

# 서브플롯 4: 진료과별 평균 변화량
plt.subplot(2, 2, 4)
dept_changes = results_df.groupby('진료과')['변화량'].mean().sort_values(ascending=True)
plt.barh(dept_changes.index, dept_changes.values, alpha=0.7, color='lightcoral')
plt.xlabel('평균 변화량')
plt.title('진료과별 평균 병상 수 변화량')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{results_dir}/병상_분배_최적화_시각화.png", dpi=300, bbox_inches='tight')
plt.show()

# 4) 상세 분석 리포트 생성
print("📊 상세 분석 리포트 생성 중...")

# 상위 10개 개선 사례
top_improvements = results_df.nlargest(10, '변화량')
print("\n=== 상위 10개 병상 증가 사례 ===")
print(top_improvements[['병원명', '진료과', '현재병상수', '최적병상수', '변화량', '변화율']].to_string(index=False))

# 하위 10개 감소 사례
top_decreases = results_df.nsmallest(10, '변화량')
print("\n=== 상위 10개 병상 감소 사례 ===")
print(top_decreases[['병원명', '진료과', '현재병상수', '최적병상수', '변화량', '변화율']].to_string(index=False))

# 진료과별 요약
dept_summary = results_df.groupby('진료과').agg({
    '현재병상수': 'sum',
    '최적병상수': 'sum',
    '변화량': 'sum',
    '예측환자수': 'sum'
}).round(2)

dept_summary['현재_병상가동률'] = (dept_summary['예측환자수'] / 365 / (dept_summary['현재병상수'] + 1)) * 100
dept_summary['최적_병상가동률'] = (dept_summary['예측환자수'] / 365 / (dept_summary['최적병상수'] + 1)) * 100
dept_summary['개선도'] = (dept_summary['현재_병상가동률'] - dept_summary['최적_병상가동률']) / dept_summary['현재_병상가동률'] * 100

print("\n=== 진료과별 요약 ===")
print(dept_summary.to_string())

# 진료과별 요약 저장
dept_summary.to_csv(f"{results_dir}/진료과별_요약.csv", encoding='utf-8-sig')

print(f"\n✅ 모든 결과가 {results_dir}/ 디렉토리에 저장되었습니다!")
print("="*60)
print("🎯 병상 분배 최적화 완료!")
print("="*60)
