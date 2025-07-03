# SLSQP 기반 전문의 분배 최적화
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize
plt.rcParams['font.family'] = 'Pretendard'
plt.rcParams['axes.unicode_minus'] = False

print("=== SLSQP 전문의 분배 최적화 모델 ===")
print("📊 SciPy SLSQP를 사용한 전문의 효율적 분배 시스템")
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
# 3) SLSQP 최적화 모델 설정
# --------------------------------------------------
print("3/6: SLSQP 최적화 모델 설정 중...")

# 초기값 설정
initial_doctors = current_df['현재전문의수'].values
patients_array = current_df['예측환자수'].values

# 경계 설정 (현재의 60%~140% 범위)
bounds = []
for d in initial_doctors:
    lower = max(1, int(d * 0.6))
    upper = max(lower + 1, int(d * 1.4))
    bounds.append((lower, upper))

# 목적 함수: 환자당 전문의 비율의 표준편차 최소화
def objective_function(doctors, patients):
    ratios = patients / (doctors + 1)  # 0으로 나누기 방지
    return np.std(ratios)

# 제약조건: 총 전문의 수는 현재와 동일
def constraint_total_doctors(doctors):
    return np.sum(doctors) - total_doctors

constraints = [
    {'type': 'eq', 'fun': constraint_total_doctors}
]

print(f"✅ SLSQP 최적화 모델 설정 완료")
print(f"  - 의사결정 변수: {len(initial_doctors)}개")
print(f"  - 제약조건: 총 전문의 수 유지")
print()

# --------------------------------------------------
# 4) SLSQP 최적화 실행
# --------------------------------------------------
print("4/6: SLSQP 최적화 실행 중...")

# 시드 고정
np.random.seed(42)

# 최적화 실행
result = minimize(
    lambda x: objective_function(x, patients_array), 
    initial_doctors, 
    method='SLSQP', 
    bounds=bounds, 
    constraints=constraints, 
    options={'maxiter': 1000}
)

print(f"✅ SLSQP 최적화 완료")
print(f"  - 최적화 성공: {result.success}")
print(f"  - 반복 횟수: {result.nit}")
print(f"  - 목적 함수 값: {result.fun:.4f}")
print()

# --------------------------------------------------
# 5) 결과 분석 및 저장
# --------------------------------------------------
print("5/6: 결과 분석 및 저장 중...")

results = []
for idx, row in current_df.iterrows():
    병원명 = row['병원명']
    진료과 = row['진료과']
    최적전문의수 = result.x[idx]
    현재전문의수 = row['현재전문의수']
    예측환자수 = row['예측환자수']
    변화량 = 최적전문의수 - 현재전문의수
    변화율 = (변화량 / 현재전문의수 * 100) if 현재전문의수 != 0 else 0
    현재_환자당전문의비율 = 예측환자수 / (현재전문의수 + 1)
    최적_환자당전문의비율 = 예측환자수 / (최적전문의수 + 1)
    
    results.append({
        '병원명': 병원명,
        '진료과': 진료과,
        '현재전문의수': 현재전문의수,
        '최적전문의수': 최적전문의수,
        '변화량': 변화량,
        '변화율': 변화율,
        '예측환자수': 예측환자수,
        '현재_환자당전문의비율': 현재_환자당전문의비율,
        '최적_환자당전문의비율': 최적_환자당전문의비율
    })

results_df = pd.DataFrame(results)

# 결과 저장
output_dir = 'optimization_results_전문의_분배_최적화'
os.makedirs(output_dir, exist_ok=True)
results_df.to_csv(f'{output_dir}/전문의_분배_최적화_결과_SLSQP.csv', index=False, encoding='utf-8-sig')

print(f"✅ SLSQP 결과 저장 완료: {output_dir}/전문의_분배_최적화_결과_SLSQP.csv")

# --------------------------------------------------
# 6) 시각화
# --------------------------------------------------
print("6/6: 시각화 생성 중...")

plt.figure(figsize=(15, 10))

# 서브플롯 1: 현재 vs 최적 전문의 수 비교
plt.subplot(2, 3, 1)
plt.scatter(results_df['현재전문의수'], results_df['최적전문의수'], alpha=0.7, s=100)
max_doctors = max(results_df['현재전문의수'].max(), results_df['최적전문의수'].max())
plt.plot([0, max_doctors], [0, max_doctors], 'r--', alpha=0.5)
plt.xlabel('현재 전문의 수')
plt.ylabel('최적 전문의 수')
plt.title('현재 vs 최적 전문의 수 (SLSQP)')
plt.grid(True, alpha=0.3)

# 서브플롯 2: 전문의 변화량
plt.subplot(2, 3, 2)
colors = ['red' if x < 0 else 'blue' if x > 0 else 'gray' for x in results_df['변화량']]
plt.barh(range(len(results_df)), results_df['변화량'], color=colors, alpha=0.7)
plt.xlabel('전문의 수 변화량')
plt.title('병원-진료과별 전문의 수 변화량 (SLSQP)')
plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
plt.yticks(range(len(results_df)), [f"{row['병원명']}-{row['진료과']}" for _, row in results_df.iterrows()], fontsize=8)
plt.grid(True, alpha=0.3)

# 서브플롯 3: 환자당 전문의 비율 비교
plt.subplot(2, 3, 3)
x = np.arange(len(results_df))
width = 0.35
plt.bar(x - width/2, results_df['현재_환자당전문의비율'], width, label='현재', alpha=0.7)
plt.bar(x + width/2, results_df['최적_환자당전문의비율'], width, label='최적', alpha=0.7)
plt.xlabel('병원-진료과')
plt.ylabel('환자당 전문의 비율')
plt.title('현재 vs 최적 환자당 전문의 비율 (SLSQP)')
plt.xticks(x, [f"{row['병원명']}-{row['진료과']}" for _, row in results_df.iterrows()], rotation=45, fontsize=8)
plt.legend()
plt.grid(True, alpha=0.3)

# 서브플롯 4: 비율 개선도
plt.subplot(2, 3, 4)
개선도 = results_df['최적_환자당전문의비율'] - results_df['현재_환자당전문의비율']
colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in 개선도]
plt.barh(range(len(results_df)), 개선도, color=colors, alpha=0.7)
plt.xlabel('환자당 전문의 비율 개선도')
plt.title('병원-진료과별 비율 개선도 (SLSQP)')
plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
plt.yticks(range(len(results_df)), [f"{row['병원명']}-{row['진료과']}" for _, row in results_df.iterrows()], fontsize=8)
plt.grid(True, alpha=0.3)

# 서브플롯 5: 현재 vs 최적 비율 산점도
plt.subplot(2, 3, 5)
plt.scatter(results_df['현재_환자당전문의비율'], results_df['최적_환자당전문의비율'], 
           alpha=0.7, s=100, c=results_df['변화량'], cmap='RdYlBu')
plt.colorbar(label='변화량')
max_ratio = max(results_df['현재_환자당전문의비율'].max(), results_df['최적_환자당전문의비율'].max())
plt.plot([0, max_ratio], [0, max_ratio], 'r--', alpha=0.5)
plt.xlabel('현재 환자당 전문의 비율')
plt.ylabel('최적 환자당 전문의 비율')
plt.title('현재 vs 최적 비율 비교 (SLSQP)')
plt.grid(True, alpha=0.3)

# 서브플롯 6: 비율 분포 비교
plt.subplot(2, 3, 6)
plt.hist([results_df['현재_환자당전문의비율'], results_df['최적_환자당전문의비율']], 
         label=['현재', '최적'], alpha=0.7, bins=10)
plt.xlabel('환자당 전문의 비율')
plt.ylabel('빈도')
plt.title('비율 분포 비교 (SLSQP)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/전문의_분배_최적화_시각화_SLSQP.png', dpi=300, bbox_inches='tight')
plt.show()

# 성능 지표 계산
현재_비율_표준편차 = results_df['현재_환자당전문의비율'].std()
최적_비율_표준편차 = results_df['최적_환자당전문의비율'].std()
개선도 = (현재_비율_표준편차 - 최적_비율_표준편차) / 현재_비율_표준편차 * 100

print("\n=== SLSQP 최적화 결과 요약 ===")
print(results_df[['병원명', '진료과', '현재전문의수', '최적전문의수', '변화량', '변화율', '현재_환자당전문의비율', '최적_환자당전문의비율']].round(2).to_string(index=False))

print(f"\n📊 성능 지표:")
print(f"  - 현재 비율 표준편차: {현재_비율_표준편차:.2f}")
print(f"  - 최적 비율 표준편차: {최적_비율_표준편차:.2f}")
print(f"  - 비율 개선도: {개선도:.1f}%")
print(f"  - 최적화 성공 여부: {result.success}")
print(f"  - 반복 횟수: {result.nit}")

print(f"\n✅ 모든 결과가 {output_dir}/ 디렉토리에 저장되었습니다!")
print("="*60)
print("🎯 SLSQP 전문의 분배 최적화 완료!")
print("="*60) 