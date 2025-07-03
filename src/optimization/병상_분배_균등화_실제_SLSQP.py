# SLSQP 기반 병상 분배 균등화 최적화
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize, NonlinearConstraint

plt.rcParams['font.family'] = 'Pretendard'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
# ... (기존과 동일)
df_hospital = pd.read_csv('new_merged_data/병원_통합_데이터_호스피스 삭제.csv')
df_pred = pd.read_csv('analysis_data/병원별_진료과별_입원_미래3년_예측결과.csv')
df_pred = df_pred[df_pred['예측연도'] == 2024]
df_pred['병원명'] = df_pred['병원'].astype(str).str.strip()
df_hospital['병원명'] = df_hospital['병원명'].astype(str).str.strip()
df_pred['병원명'] = df_pred['병원명'].replace('중앙', '서울')
bed_columns = [col for col in df_hospital.columns if not col.endswith('_전문의수') and col != '병원명']
df_hospital['총병상수'] = df_hospital[bed_columns].sum(axis=1)
hospital_patients = df_pred.groupby('병원명')['XGB예측'].sum().reset_index()
hospital_patients.columns = ['병원명', '총예측환자수']
data = []
total_beds = 0
for idx, row in hospital_patients.iterrows():
    병원 = row['병원명']
    예측환자수 = row['총예측환자수']
    hosp_row = df_hospital[df_hospital['병원명'] == 병원]
    if len(hosp_row) > 0:
        현재병상수 = float(hosp_row['총병상수'].iloc[0])
        if pd.notnull(현재병상수) and 현재병상수 > 0:
            data.append({'병원명': 병원, '현재병상수': 현재병상수, '예측환자수': 예측환자수})
            total_beds += 현재병상수
current_df = pd.DataFrame(data)

# 최적화 함수 정의

def calculate_utilization(beds, patients):
    return (patients / 365) / (beds + 1) * 100

def objective_function(beds_array, patients_array):
    utilizations = [calculate_utilization(b, p) for b, p in zip(beds_array, patients_array)]
    return np.std(utilizations)

def constraint_total_beds(beds_array, total_beds):
    return np.sum(beds_array) - total_beds

def constraint_min_utilization(beds_array, patients_array):
    return np.array([calculate_utilization(b, p) for b, p in zip(beds_array, patients_array)]) - 40

def constraint_max_utilization(beds_array, patients_array):
    return 90 - np.array([calculate_utilization(b, p) for b, p in zip(beds_array, patients_array)])

initial_beds = current_df['현재병상수'].values
patients_array = current_df['예측환자수'].values
bounds = [(max(1, int(b*0.6)), int(b*1.4)) for b in initial_beds]
constraints = [
    NonlinearConstraint(lambda x: constraint_total_beds(x, total_beds), 0, 0),
    NonlinearConstraint(lambda x: constraint_min_utilization(x, patients_array), 0, np.inf),
    NonlinearConstraint(lambda x: constraint_max_utilization(x, patients_array), 0, np.inf)
]
result = minimize(lambda x: objective_function(x, patients_array), initial_beds, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 1000})

# 결과 분석 및 저장
print("결과 분석 및 저장 중...")

results = []
for idx, row in current_df.iterrows():
    병원명 = row['병원명']
    최적병상수 = result.x[idx]
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

# 결과 저장
output_dir = 'optimization_results_병상_분배_균등화_실제'
os.makedirs(output_dir, exist_ok=True)
results_df.to_csv(f'{output_dir}/병상_분배_균등화_결과_SLSQP.csv', index=False, encoding='utf-8-sig')

print(f"✅ SLSQP 결과 저장 완료: {output_dir}/병상_분배_균등화_결과_SLSQP.csv")

# 시각화
plt.figure(figsize=(15, 10))

# 서브플롯 1: 현재 vs 최적 병상 수 비교
plt.subplot(2, 3, 1)
plt.scatter(results_df['현재병상수'], results_df['최적병상수'], alpha=0.7, s=100)
max_beds = max(results_df['현재병상수'].max(), results_df['최적병상수'].max())
plt.plot([0, max_beds], [0, max_beds], 'r--', alpha=0.5)
plt.xlabel('현재 병상 수')
plt.ylabel('최적 병상 수')
plt.title('현재 vs 최적 병상 수 (SLSQP)')
plt.grid(True, alpha=0.3)

# 서브플롯 2: 병상 변화량
plt.subplot(2, 3, 2)
colors = ['red' if x < 0 else 'blue' if x > 0 else 'gray' for x in results_df['변화량']]
plt.barh(results_df['병원명'], results_df['변화량'], color=colors, alpha=0.7)
plt.xlabel('병상 수 변화량')
plt.title('병원별 병상 수 변화량 (SLSQP)')
plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
plt.grid(True, alpha=0.3)

# 서브플롯 3: 가동률 비교
plt.subplot(2, 3, 3)
x = np.arange(len(results_df))
width = 0.35
plt.bar(x - width/2, results_df['현재_병상가동률'], width, label='현재', alpha=0.7)
plt.bar(x + width/2, results_df['최적_병상가동률'], width, label='최적', alpha=0.7)
plt.axhline(y=65, color='red', linestyle='--', alpha=0.7, label='목표(65%)')
plt.xlabel('병원')
plt.ylabel('병상가동률 (%)')
plt.title('현재 vs 최적 병상가동률 (SLSQP)')
plt.xticks(x, list(results_df['병원명']), rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# 서브플롯 4: 가동률 개선도
plt.subplot(2, 3, 4)
개선도 = results_df['최적_병상가동률'] - results_df['현재_병상가동률']
colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in 개선도]
plt.barh(results_df['병원명'], 개선도, color=colors, alpha=0.7)
plt.xlabel('가동률 개선도 (%)')
plt.title('병원별 가동률 개선도 (SLSQP)')
plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
plt.grid(True, alpha=0.3)

# 서브플롯 5: 현재 vs 최적 가동률 산점도
plt.subplot(2, 3, 5)
plt.scatter(results_df['현재_병상가동률'], results_df['최적_병상가동률'], 
           alpha=0.7, s=100, c=results_df['변화량'], cmap='RdYlBu')
plt.colorbar(label='변화량')
max_util = max(results_df['현재_병상가동률'].max(), results_df['최적_병상가동률'].max())
plt.plot([0, max_util], [0, max_util], 'r--', alpha=0.5)
plt.axhline(y=65, color='red', linestyle='--', alpha=0.7, label='목표(65%)')
plt.xlabel('현재 병상가동률 (%)')
plt.ylabel('최적 병상가동률 (%)')
plt.title('현재 vs 최적 가동률 비교 (SLSQP)')
plt.legend()
plt.grid(True, alpha=0.3)

# 서브플롯 6: 가동률 분포 비교
plt.subplot(2, 3, 6)
plt.hist([results_df['현재_병상가동률'], results_df['최적_병상가동률']], 
         label=['현재', '최적'], alpha=0.7, bins=10)
plt.axvline(x=65, color='red', linestyle='--', alpha=0.7, label='목표(65%)')
plt.xlabel('병상가동률 (%)')
plt.ylabel('병원 수')
plt.title('가동률 분포 비교 (SLSQP)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/병상_분배_균등화_시각화_SLSQP.png', dpi=300, bbox_inches='tight')
plt.show()

# 성능 지표 계산
현재_가동률_표준편차 = results_df['현재_병상가동률'].std()
최적_가동률_표준편차 = results_df['최적_병상가동률'].std()
가동률_개선도 = (현재_가동률_표준편차 - 최적_가동률_표준편차) / 현재_가동률_표준편차 * 100

print("\n=== SLSQP 최적화 결과 요약 ===")
print(results_df[['병원명', '현재병상수', '최적병상수', '변화량', '변화율', '현재_병상가동률', '최적_병상가동률']].round(2).to_string(index=False))

print(f"\n📊 성능 지표:")
print(f"  - 현재 가동률 표준편차: {현재_가동률_표준편차:.2f}%")
print(f"  - 최적 가동률 표준편차: {최적_가동률_표준편차:.2f}%")
print(f"  - 가동률 개선도: {가동률_개선도:.1f}%")
print(f"  - 최적화 성공 여부: {result.success}")
print(f"  - 반복 횟수: {result.nit}")

print(f"\n✅ 모든 결과가 {output_dir}/ 디렉토리에 저장되었습니다!")
print("="*60)
print("🎯 SLSQP 병상 분배 균등화 최적화 완료!")
print("="*60) 