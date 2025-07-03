import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import pulp
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpStatus, value, PULP_CBC_CMD
import matplotlib.pyplot as plt
from datetime import datetime
import os

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Pretendard'
plt.rcParams['axes.unicode_minus'] = False

print("=== 병상 분배 균등화 최적화 모델 (실제적 접근) ===")
print("📊 가동률 균등화를 위한 현실적 최적화 시스템")
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
# 2) 실제적 최적화 모델 설정
# --------------------------------------------------
print("2/5: 실제적 최적화 모델 설정 중...")

# 목표 가동률 계산 (전체 평균)
target_utilization = 65.0  # 목표 가동률 65%

prob = LpProblem("병상_분배_균등화_실제", LpMinimize)

# 의사결정 변수: 각 병원별 병상 수
beds = {}
for idx, row in current_df.iterrows():
    병원명 = row['병원명']
    beds[병원명] = LpVariable(f"beds_{병원명}", lowBound=1, cat='Integer')

# 목적 함수: 가동률 균등화를 위한 단순화된 접근
# 각 병원의 병상 수가 예측 환자 수에 비례하도록 조정
prob += lpSum([beds[병원명] for 병원명 in beds.keys()])

# 제약조건 1: 총 병상 수는 현재와 동일
prob += lpSum([beds[병원명] for 병원명 in beds.keys()]) == total_beds

# 제약조건 2: 각 병원별 현실적 제약 (현재의 60-140% 범위로 확대)
for idx, row in current_df.iterrows():
    병원명 = row['병원명']
    현재병상수 = row['현재병상수']
    최소병상수 = max(1, int(현재병상수 * 0.6))  # 현재의 60% 이상
    최대병상수 = int(현재병상수 * 1.4)  # 현재의 140% 이하
    prob += beds[병원명] >= 최소병상수
    prob += beds[병원명] <= 최대병상수

# 제약조건 3: 가동률 범위 제약 (선형화)
for idx, row in current_df.iterrows():
    병원명 = row['병원명']
    일평균환자수 = row['예측환자수'] / 365
    
    # 최소 가동률 보장 (40% 이상): 일평균환자수 * 100 >= 40 * beds
    prob += 일평균환자수 * 100 >= 40 * beds[병원명]
    
    # 최대 가동률 제한 (90% 이하): 일평균환자수 * 100 <= 90 * beds
    prob += 일평균환자수 * 100 <= 90 * beds[병원명]

# 제약조건 4: 가동률 편차 정의 (선형화된 근사)
# 목표 가동률 65%에 가까워지도록 제약
for idx, row in current_df.iterrows():
    병원명 = row['병원명']
    일평균환자수 = row['예측환자수'] / 365
    
    # 목표 가동률 65%에 근접하도록 제약 (60-70% 범위)
    prob += 일평균환자수 * 100 >= 60 * beds[병원명]  # 최소 60%
    prob += 일평균환자수 * 100 <= 70 * beds[병원명]  # 최대 70%

print(f"✅ 실제적 최적화 모델 설정 완료 (병원별 변수 {len(beds)}개)")
print(f"  - 제약조건: 현재 병상수의 60-140% 범위")
print(f"  - 목표: 가동률 {target_utilization}%에 근접하도록 균등화")
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

results_dir = "optimization_results_병상_분배_균등화_실제"
os.makedirs(results_dir, exist_ok=True)
results_df.to_csv(f"{results_dir}/병상_분배_균등화_결과.csv", index=False, encoding='utf-8-sig')

print(f"✅ 결과 저장 완료: {results_dir}/병상_분배_균등화_결과.csv")
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
plt.title('현재 vs 최적 병상 수 (실제적 접근)')
plt.grid(True, alpha=0.3)

# 서브플롯 2: 병상 변화량
plt.subplot(2, 3, 2)
colors = ['red' if x < 0 else 'blue' if x > 0 else 'gray' for x in results_df['변화량']]
plt.barh(results_df['병원명'], results_df['변화량'], color=colors, alpha=0.7)
plt.xlabel('병상 수 변화량')
plt.title('병원별 병상 수 변화량 (실제적 접근)')
plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
plt.grid(True, alpha=0.3)

# 서브플롯 3: 가동률 비교
plt.subplot(2, 3, 3)
x = np.arange(len(results_df))
width = 0.35
plt.bar(x - width/2, results_df['현재_병상가동률'], width, label='현재', alpha=0.7)
plt.bar(x + width/2, results_df['최적_병상가동률'], width, label='최적', alpha=0.7)
plt.axhline(y=target_utilization, color='red', linestyle='--', alpha=0.7, label=f'목표({target_utilization}%)')
plt.xlabel('병원')
plt.ylabel('병상가동률 (%)')
plt.title('현재 vs 최적 병상가동률 (실제적 접근)')
plt.xticks(x, list(results_df['병원명']), rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# 서브플롯 4: 가동률 개선도
plt.subplot(2, 3, 4)
개선도 = results_df['최적_병상가동률'] - results_df['현재_병상가동률']
colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in 개선도]
plt.barh(results_df['병원명'], 개선도, color=colors, alpha=0.7)
plt.xlabel('가동률 개선도 (%)')
plt.title('병원별 가동률 개선도 (실제적 접근)')
plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
plt.grid(True, alpha=0.3)

# 서브플롯 5: 현재 vs 최적 가동률 산점도
plt.subplot(2, 3, 5)
plt.scatter(results_df['현재_병상가동률'], results_df['최적_병상가동률'], 
           alpha=0.7, s=100, c=results_df['변화량'], cmap='RdYlBu')
plt.colorbar(label='변화량')
max_util = max(results_df['현재_병상가동률'].max(), results_df['최적_병상가동률'].max())
plt.plot([0, max_util], [0, max_util], 'r--', alpha=0.5)
plt.axhline(y=target_utilization, color='red', linestyle='--', alpha=0.7, label=f'목표({target_utilization}%)')
plt.xlabel('현재 병상가동률 (%)')
plt.ylabel('최적 병상가동률 (%)')
plt.title('현재 vs 최적 가동률 비교 (실제적 접근)')
plt.legend()
plt.grid(True, alpha=0.3)

# 서브플롯 6: 가동률 분포 비교
plt.subplot(2, 3, 6)
plt.hist([results_df['현재_병상가동률'], results_df['최적_병상가동률']], 
         label=['현재', '최적'], alpha=0.7, bins=10)
plt.axvline(x=target_utilization, color='red', linestyle='--', alpha=0.7, label=f'목표({target_utilization}%)')
plt.xlabel('병상가동률 (%)')
plt.ylabel('병원 수')
plt.title('가동률 분포 비교 (실제적 접근)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{results_dir}/병상_분배_균등화_시각화.png", dpi=300, bbox_inches='tight')
plt.show()

# 성능 지표 계산
현재_가동률_표준편차 = results_df['현재_병상가동률'].std()
최적_가동률_표준편차 = results_df['최적_병상가동률'].std()
가동률_개선도 = (현재_가동률_표준편차 - 최적_가동률_표준편차) / 현재_가동률_표준편차 * 100

print("\n=== 병원별 최적화 결과 요약 (실제적 접근) ===")
print(results_df[['병원명', '현재병상수', '최적병상수', '변화량', '변화율', '현재_병상가동률', '최적_병상가동률']].round(2).to_string(index=False))

print(f"\n📊 성능 지표:")
print(f"  - 현재 가동률 표준편차: {현재_가동률_표준편차:.2f}%")
print(f"  - 최적 가동률 표준편차: {최적_가동률_표준편차:.2f}%")
print(f"  - 가동률 개선도: {가동률_개선도:.1f}%")
print(f"  - 목표 가동률: {target_utilization}%")

print(f"\n✅ 모든 결과가 {results_dir}/ 디렉토리에 저장되었습니다!")
print("="*60)
print("🎯 병상 분배 균등화 최적화(실제적 접근) 완료!")
print("="*60) 