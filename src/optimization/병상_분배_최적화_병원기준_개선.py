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

print("=== 병상 분배 최적화 모델 (병원 단위) - 개선본 ===")
print("📊 가동률 균등화 및 현실적 제약을 고려한 병상 분배 시스템")
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

# 병원명 컬럼명 통일 및 매칭 수정
df_pred['병원명'] = df_pred['병원'].astype(str).str.strip()
df_hospital['병원명'] = df_hospital['병원명'].astype(str).str.strip()

# 병원명 매칭 수정: "중앙" → "서울"
df_pred['병원명'] = df_pred['병원명'].replace('중앙', '서울')

print(f"예측 데이터 병원명: {list(df_pred['병원명'].unique())}")
print(f"병원 데이터 병원명: {list(df_hospital['병원명'].unique())}")

# 병상 관련 컬럼 추출 (전문의수로 끝나지 않는 컬럼들)
bed_columns = [col for col in df_hospital.columns if not col.endswith('_전문의수') and col != '병원명']
df_hospital['총병상수'] = df_hospital[bed_columns].sum(axis=1)

# 병원별 예측 환자 수 집계
hospital_patients = df_pred.groupby('병원명')['XGB예측'].sum().reset_index()
hospital_patients.columns = ['병원명', '총예측환자수']

print(f"병원별 예측 환자 수:")
print(hospital_patients)

# 병원별 현재 상황 분석
data = []
total_beds = 0
total_patients = 0

for idx, row in hospital_patients.iterrows():
    병원 = row['병원명']
    예측환자수 = row['총예측환자수']
    hosp_row = df_hospital[df_hospital['병원명'] == 병원]
    
    if len(hosp_row) > 0:
        현재병상수 = float(hosp_row['총병상수'].iloc[0])
        if pd.notnull(현재병상수) and 현재병상수 > 0:
            data.append({
                '병원명': 병원,
                '현재병상수': 현재병상수,
                '예측환자수': 예측환자수
            })
            total_beds += 현재병상수
            total_patients += 예측환자수
            print(f"✅ {병원}: 병상 {현재병상수}개, 예측환자 {예측환자수:.0f}명")
        else:
            print(f"⚠️ {병원}: 병상 데이터 없음")
    else:
        print(f"❌ {병원}: 병원 데이터 없음")

current_df = pd.DataFrame(data)

print(f"\n✅ 데이터 로드 및 집계 완료: 병원 수 {len(current_df)}개")
print(f"  - 총 병상 수: {total_beds:.0f}개")
print(f"  - 총 예측 환자 수: {total_patients:.0f}명")
print()

# --------------------------------------------------
# 2) 개선된 최적화 모델 설정
# --------------------------------------------------
print("2/5: 개선된 최적화 모델 설정 중...")

prob = LpProblem("병원별_병상_분배_최적화_개선", LpMinimize)

# 의사결정 변수: 각 병원별 병상 수
beds = {}
for idx, row in current_df.iterrows():
    병원명 = row['병원명']
    beds[병원명] = LpVariable(f"beds_{병원명}", lowBound=1, cat='Integer')

# 목적 함수: 병상 수 변화량의 절댓값 합 최소화 (현재 상태에서 최소한의 변화)
# 이를 위해 양수/음수 변화량 변수 사용
bed_change_positive = {}
bed_change_negative = {}

for idx, row in current_df.iterrows():
    병원명 = row['병원명']
    현재병상수 = row['현재병상수']
    bed_change_positive[병원명] = LpVariable(f"change_pos_{병원명}", lowBound=0)
    bed_change_negative[병원명] = LpVariable(f"change_neg_{병원명}", lowBound=0)

# 목적 함수: 변화량의 합 최소화
prob += lpSum([bed_change_positive[병원명] + bed_change_negative[병원명] 
               for 병원명 in beds.keys()])

# 제약조건 1: 총 병상 수는 현재와 동일
prob += lpSum([beds[병원명] for 병원명 in beds.keys()]) == total_beds

# 제약조건 2: 각 병원별 현실적 제약 (현재의 80-120% 범위)
for idx, row in current_df.iterrows():
    병원명 = row['병원명']
    현재병상수 = row['현재병상수']
    최소병상수 = max(1, int(현재병상수 * 0.8))  # 현재의 80% 이상
    최대병상수 = int(현재병상수 * 1.2)  # 현재의 120% 이하
    prob += beds[병원명] >= 최소병상수
    prob += beds[병원명] <= 최대병상수

# 제약조건 3: 변화량 정의
for idx, row in current_df.iterrows():
    병원명 = row['병원명']
    현재병상수 = row['현재병상수']
    
    # 병상수 = 현재병상수 + 양수변화량 - 음수변화량
    prob += beds[병원명] == 현재병상수 + bed_change_positive[병원명] - bed_change_negative[병원명]

print(f"✅ 개선된 최적화 모델 설정 완료 (병원별 변수 {len(beds)}개)")
print(f"  - 제약조건: 현재 병상수의 80-120% 범위")
print(f"  - 목표: 최소한의 변화로 균형 조정")
print()

# --------------------------------------------------
# 3) 최적화 실행
# --------------------------------------------------
print("3/5: 최적화 실행 중...")
try:
    prob.solve(PULP_CBC_CMD(msg=False))
    print("✅ CBC 솔버 최적화 완료!")
except Exception as e:
    print(f"⚠️ CBC 솔버 오류: {e}")
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

results_dir = "optimization_results_병상_분배_최적화_병원기준_개선"
os.makedirs(results_dir, exist_ok=True)
results_df.to_csv(f"{results_dir}/병상_분배_최적화_결과.csv", index=False, encoding='utf-8-sig')

print(f"✅ 결과 저장 완료: {results_dir}/병상_분배_최적화_결과.csv")
print()

# --------------------------------------------------
# 5) 시각화 및 리포트
# --------------------------------------------------
print("5/5: 시각화 및 리포트 생성 중...")

plt.figure(figsize=(15, 10))

# 서브플롯 1: 현재 vs 최적 병상 수 비교
plt.subplot(2, 3, 1)
plt.scatter(results_df['현재병상수'], results_df['최적병상수'], alpha=0.7, s=100)
max_beds = max(results_df['현재병상수'].max(), results_df['최적병상수'].max())
plt.plot([0, max_beds], [0, max_beds], 'r--', alpha=0.5)
plt.xlabel('현재 병상 수')
plt.ylabel('최적 병상 수')
plt.title('현재 vs 최적 병상 수 (병원별)')
plt.grid(True, alpha=0.3)

# 서브플롯 2: 병상 변화량
plt.subplot(2, 3, 2)
colors = ['red' if x < 0 else 'blue' if x > 0 else 'gray' for x in results_df['변화량']]
plt.barh(results_df['병원명'], results_df['변화량'], color=colors, alpha=0.7)
plt.xlabel('병상 수 변화량')
plt.title('병원별 병상 수 변화량')
plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
plt.grid(True, alpha=0.3)

# 서브플롯 3: 가동률 비교
plt.subplot(2, 3, 3)
x = np.arange(len(results_df))
width = 0.35
plt.bar(x - width/2, results_df['현재_병상가동률'], width, label='현재', alpha=0.7)
plt.bar(x + width/2, results_df['최적_병상가동률'], width, label='최적', alpha=0.7)
plt.xlabel('병원')
plt.ylabel('병상가동률 (%)')
plt.title('현재 vs 최적 병상가동률')
plt.xticks(x, list(results_df['병원명']), rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# 서브플롯 4: 가동률 개선도
plt.subplot(2, 3, 4)
개선도 = results_df['최적_병상가동률'] - results_df['현재_병상가동률']
colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in 개선도]
plt.barh(results_df['병원명'], 개선도, color=colors, alpha=0.7)
plt.xlabel('가동률 개선도 (%)')
plt.title('병원별 가동률 개선도')
plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
plt.grid(True, alpha=0.3)

# 서브플롯 5: 현재 vs 최적 가동률 산점도
plt.subplot(2, 3, 5)
plt.scatter(results_df['현재_병상가동률'], results_df['최적_병상가동률'], 
           alpha=0.7, s=100, c=results_df['변화량'], cmap='RdYlBu')
plt.colorbar(label='변화량')
max_util = max(results_df['현재_병상가동률'].max(), results_df['최적_병상가동률'].max())
plt.plot([0, max_util], [0, max_util], 'r--', alpha=0.5)
plt.xlabel('현재 병상가동률 (%)')
plt.ylabel('최적 병상가동률 (%)')
plt.title('현재 vs 최적 가동률 비교')
plt.grid(True, alpha=0.3)

# 서브플롯 6: 병원별 예측 환자 수
plt.subplot(2, 3, 6)
plt.barh(results_df['병원명'], results_df['예측환자수'], alpha=0.7, color='orange')
plt.xlabel('예측 환자 수')
plt.title('병원별 예측 환자 수')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{results_dir}/병상_분배_최적화_시각화.png", dpi=300, bbox_inches='tight')
plt.show()

# 성능 지표 계산
현재_가동률_표준편차 = results_df['현재_병상가동률'].std()
최적_가동률_표준편차 = results_df['최적_병상가동률'].std()
가동률_개선도 = (현재_가동률_표준편차 - 최적_가동률_표준편차) / 현재_가동률_표준편차 * 100

print("\n=== 병원별 최적화 결과 요약 (개선본) ===")
print(results_df[['병원명', '현재병상수', '최적병상수', '변화량', '변화율', '현재_병상가동률', '최적_병상가동률']].round(2).to_string(index=False))

print(f"\n📊 성능 지표:")
print(f"  - 현재 가동률 표준편차: {현재_가동률_표준편차:.2f}%")
print(f"  - 최적 가동률 표준편차: {최적_가동률_표준편차:.2f}%")
print(f"  - 가동률 개선도: {가동률_개선도:.1f}%")

print(f"\n✅ 모든 결과가 {results_dir}/ 디렉토리에 저장되었습니다!")
print("="*60)
print("🎯 병상 분배 최적화(병원 단위, 개선본) 완료!")
print("="*60) 