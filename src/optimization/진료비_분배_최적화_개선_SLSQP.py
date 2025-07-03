import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from datetime import datetime
import json
import os
from scipy.optimize import minimize

plt.rcParams['font.family'] = 'Pretendard'
plt.rcParams['axes.unicode_minus'] = False

print("=== 진료비 분배 최적화 모델 (SLSQP 개선본) ===")
print("📊 상병코드별 평균진료비 활용한 정확한 추정 시스템 (SLSQP)")
print()

# --------------------------------------------------
# 1) 데이터 로드 및 전처리
# --------------------------------------------------
print("1/6: 데이터 로드 및 전처리 중...")

df_cost = pd.read_csv('new_merged_data/df_result2_with_심평원_진료비.csv')
df_demand = pd.read_csv('analysis_data/병원별_진료과별_미래3년_예측결과.csv')
df_hospital = pd.read_csv('new_merged_data/병원_통합_데이터_호스피스 삭제.csv')
df_avg_cost = pd.read_csv('new_merged_data/상병코드별_전체_평균진료비.csv')

print(f"✅ 진료비 데이터: {len(df_cost)}개 레코드")
print(f"✅ 수요예측 데이터: {len(df_demand)}개 레코드")
print(f"✅ 병원 통합 데이터: {len(df_hospital)}개 병원")
print(f"✅ 상병코드별 평균진료비: {len(df_avg_cost)}개 상병코드")
print()

# --------------------------------------------------
# 2) 데이터 전처리 및 진료비 추정 개선
# --------------------------------------------------
print("2/6: 데이터 전처리 및 진료비 추정 개선 중...")

df_demand['병원명'] = df_demand['병원'].replace('중앙', '서울')
df_demand_2024 = df_demand[df_demand['예측연도'] == 2024].copy()
df_demand_2024['예측환자수'] = df_demand_2024['ARIMA예측']
df_cost['진료비(천원)'] = pd.to_numeric(df_cost['진료비(천원)'], errors='coerce')
df_avg_cost['평균진료비_천원'] = df_avg_cost['평균요양급여비용총액'] / 1000
avg_cost_dict = dict(zip(df_avg_cost['주상병코드'], df_avg_cost['평균진료비_천원']))

print(f"✅ 상병코드별 평균진료비 매핑 완료: {len(avg_cost_dict)}개 상병코드")
print()

def estimate_missing_cost(row, avg_cost_dict):
    if pd.notna(row['진료비(천원)']) and row['진료비(천원)'] > 0:
        return row['진료비(천원)']
    같은_상병 = df_cost[(df_cost['상병코드'] == row['상병코드']) & (df_cost['진료비(천원)'].notna()) & (df_cost['진료비(천원)'] > 0)]
    if len(같은_상병) > 0:
        평균_인원당_진료비 = 같은_상병['진료비(천원)'].sum() / 같은_상병['연인원'].sum()
        return row['연인원'] * 평균_인원당_진료비
    if row['상병코드'] in avg_cost_dict:
        return row['연인원'] * avg_cost_dict[row['상병코드']]
    같은_진료과 = df_cost[(df_cost['진료과'] == row['진료과']) & (df_cost['진료비(천원)'].notna()) & (df_cost['진료비(천원)'] > 0)]
    if len(같은_진료과) > 0:
        평균_인원당_진료비 = 같은_진료과['진료비(천원)'].sum() / 같은_진료과['연인원'].sum()
        return row['연인원'] * 평균_인원당_진료비
    전체_평균 = df_cost[df_cost['진료비(천원)'].notna()]['진료비(천원)'].mean()
    return row['연인원'] * (전체_평균 / df_cost['연인원'].mean())

print("빈 진료비 값 추정 중...")
추정_완료 = 0
추정_실패 = 0
for idx, row in df_cost.iterrows():
    if pd.isna(row['진료비(천원)']) or row['진료비(천원)'] == 0:
        추정값 = estimate_missing_cost(row, avg_cost_dict)
        df_cost.loc[idx, '진료비(천원)'] = 추정값
        if 추정값 > 0:
            추정_완료 += 1
        else:
            추정_실패 += 1
print(f"✅ 진료비 추정 완료: {추정_완료}개 성공, {추정_실패}개 실패")
print()

# --------------------------------------------------
# 3) 진료과별 통합 데이터 생성
# --------------------------------------------------
print("3/6: 진료과별 통합 데이터 생성 중...")

cost_by_dept = df_cost.groupby(['지역', '진료과']).agg({'연인원': 'sum', '진료비(천원)': 'sum'}).reset_index()
demand_by_dept = df_demand_2024.groupby(['병원명', '진료과']).agg({'예측환자수': 'sum'}).reset_index()
medical_staff_cols = [col for col in df_hospital.columns if '전문의수' in col]
medical_staff_data = df_hospital[['병원명'] + medical_staff_cols].copy()
dept_mapping = {
    '가정의학과_전문의수': '가정의학과',
    '내과_전문의수': '내과',
    '비뇨의학과_전문의수': '비뇨의학과',
    '산부인과_전문의수': '산부인과',
    '소아청소년과_전문의수': '소아청소년과',
    '신경과_전문의수': '신경과',
    '신경외과_전문의수': '신경외과',
    '안과_전문의수': '안과',
    '외과_전문의수': '외과',
    '응급의학과_전문의수': '응급의학과',
    '이비인후과_전문의수': '이비인후과',
    '재활의학과_전문의수': '재활의학과',
    '정신건강의학과_전문의수': '정신건강의학과',
    '정형외과_전문의수': '정형외과',
    '치과_전문의수': '치과',
    '피부과_전문의수': '피부과'
}
medical_staff_long = []
for col in medical_staff_cols:
    if col in dept_mapping:
        dept_name = dept_mapping[col]
        temp_df = medical_staff_data[['병원명', col]].copy()
        temp_df['진료과'] = dept_name
        temp_df['의사수'] = temp_df[col]
        medical_staff_long.append(temp_df[['병원명', '진료과', '의사수']])
medical_staff_combined = pd.concat(medical_staff_long, ignore_index=True)
bed_data = df_hospital[['병원명', '일반입원실_상급', '일반입원실_일반']].copy()
bed_data['총병상수'] = bed_data['일반입원실_상급'] + bed_data['일반입원실_일반']
print(f"✅ 진료과별 통합 데이터 생성 완료")
print()

# --------------------------------------------------
# 4) 성과지표 계산
# --------------------------------------------------
print("4/6: 성과지표 계산 중...")
merged_data = cost_by_dept.merge(
    demand_by_dept, 
    left_on=['지역', '진료과'], 
    right_on=['병원명', '진료과'], 
    how='outer'
).merge(
    medical_staff_combined,
    on=['병원명', '진료과'],
    how='outer'
).merge(
    bed_data[['병원명', '총병상수']].copy(),
    on='병원명',
    how='outer'
)
merged_data = merged_data.fillna(0)
merged_data['1인당_진료비'] = merged_data['진료비(천원)'] / merged_data['연인원'].replace(0, 1)
merged_data['의사당_환자수'] = merged_data['연인원'] / merged_data['의사수'].replace(0, 1)
merged_data['일평균_입원환자수'] = merged_data['연인원'] / 365
merged_data['병상가동률'] = merged_data['일평균_입원환자수'] / merged_data['총병상수'].replace(0, 1) * 100
merged_data['효율성_점수'] = (
    (1 / merged_data['1인당_진료비'].replace(0, 1)) * 0.4 +
    merged_data['의사당_환자수'] * 0.3 +
    np.minimum(merged_data['병상가동률'] / 90, 1) * 0.3
)
merged_data['수요대비_비율'] = merged_data['연인원'] / merged_data['예측환자수'].replace(0, 1)
merged_data['적절성_점수'] = np.minimum(merged_data['수요대비_비율'], 1)
merged_data['종합_성과지표'] = (
    merged_data['효율성_점수'] * 0.6 +
    merged_data['적절성_점수'] * 0.4
)
print(f"✅ 성과지표 계산 완료")
print()

# --------------------------------------------------
# 5) SLSQP 최적화
# --------------------------------------------------
print("5/6: SLSQP 최적화 실행 중...")

n = len(merged_data)
초기값 = np.ones(n)
하한 = np.full(n, 0.1)
상한 = np.full(n, 2.0)
현재_진료비 = merged_data['진료비(천원)'].values
성과지표 = merged_data['종합_성과지표'].values
총_진료비 = np.sum(현재_진료비)

# 목적함수(음수 부호: maximize)
def obj(x):
    return -np.sum(x * 성과지표)

# 제약조건: 총 진료비 한도
con1 = {'type': 'ineq', 'fun': lambda x: 총_진료비 * 1.1 - np.sum(x * 현재_진료비)}
# 제약조건: 각 진료과별 최소 진료비(50%)
def con2_i(i):
    return {'type': 'ineq', 'fun': lambda x, i=i: x[i] * 현재_진료비[i] - 현재_진료비[i] * 0.5}
cons = [con1] + [con2_i(i) for i in range(n)]

res = minimize(obj, 초기값, method='SLSQP', bounds=list(zip(하한, 상한)), constraints=cons, options={'ftol':1e-6, 'disp':True, 'maxiter':500})

print(f"✅ SLSQP 최적화 완료 - 성공: {res.success}, 반복: {res.nit}, 목적함수값: {-res.fun:.4f}")
print()

# --------------------------------------------------
# 6) 결과 분석 및 저장/시각화
# --------------------------------------------------
print("6/6: 결과 분석 및 저장/시각화 중...")

최적_배분비율 = res.x
results = []
for i, row in merged_data.iterrows():
    현재 = row['진료비(천원)']
    최적 = 현재 * 최적_배분비율[i]
    results.append({
        '병원명': row['병원명'],
        '진료과': row['진료과'],
        '현재_진료비(천원)': 현재,
        '최적_진료비(천원)': 최적,
        '변화량(천원)': 최적 - 현재,
        '변화율(%)': ((최적 - 현재) / 현재 * 100) if 현재 > 0 else 0,
        '배분비율': 최적_배분비율[i],
        '효율성_점수': row['효율성_점수'],
        '적절성_점수': row['적절성_점수'],
        '종합_성과지표': row['종합_성과지표'],
        '현재_1인당_진료비': row['1인당_진료비'],
        '현재_의사당_환자수': row['의사당_환자수'],
        '현재_병상가동률': row['병상가동률'],
        '예측환자수': row['예측환자수']
    })
results_df = pd.DataFrame(results)
output_dir = 'optimization_results_진료비_분배_최적화_개선_SLSQP'
os.makedirs(output_dir, exist_ok=True)
results_df.to_csv(f'{output_dir}/진료비_분배_최적화_결과_SLSQP.csv', index=False, encoding='utf-8-sig')
summary_stats = {
    '총_현재_진료비': float(총_진료비),
    '총_최적_진료비': float(results_df['최적_진료비(천원)'].sum()),
    '총_변화량': float(results_df['변화량(천원)'].sum()),
    '평균_변화율': float(results_df['변화율(%)'].mean()),
    '최적화_성공': bool(res.success),
    '반복횟수': int(res.nit),
    '목적함수_값': float(-res.fun),
    '추정_완료_수': int(추정_완료),
    '추정_실패_수': int(추정_실패)
}
with open(f'{output_dir}/최적화_요약_SLSQP.json', 'w', encoding='utf-8') as f:
    json.dump(summary_stats, f, ensure_ascii=False, indent=2)

# 시각화 (기존 개선본과 동일)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('진료비 분배 최적화 결과 분석 (SLSQP 개선본)', fontsize=16, fontweight='bold')
# 1) 진료과별 변화량
ax1 = axes[0, 0]
dept_changes = results_df.groupby('진료과')['변화량(천원)'].sum().sort_values()
ax1.barh(range(len(dept_changes)), dept_changes.values, alpha=0.7, color='skyblue')
ax1.set_yticks(range(len(dept_changes)))
ax1.set_yticklabels(list(dept_changes.index))
ax1.set_xlabel('변화량 (천원)')
ax1.set_title('진료과별 진료비 변화량')
ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7)
ax1.grid(True, alpha=0.3)
# 2) 병원별 변화량
ax2 = axes[0, 1]
hosp_changes = results_df.groupby('병원명')['변화량(천원)'].sum().sort_values()
ax2.barh(range(len(hosp_changes)), hosp_changes.values, alpha=0.7, color='lightgreen')
ax2.set_yticks(range(len(hosp_changes)))
ax2.set_yticklabels(list(hosp_changes.index))
ax2.set_xlabel('변화량 (천원)')
ax2.set_title('병원별 진료비 변화량')
ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
ax2.grid(True, alpha=0.3)
# 3) 성과지표 분포
ax3 = axes[0, 2]
ax3.hist(results_df['종합_성과지표'], bins=20, alpha=0.7, color='orange', edgecolor='black')
ax3.set_xlabel('종합 성과지표')
ax3.set_ylabel('빈도')
ax3.set_title('성과지표 분포')
ax3.grid(True, alpha=0.3)
# 4) 효율성 vs 적절성
ax4 = axes[1, 0]
scatter = ax4.scatter(results_df['효율성_점수'], results_df['적절성_점수'], c=results_df['변화율(%)'], cmap='RdYlBu', alpha=0.7, s=50)
ax4.set_xlabel('효율성 점수')
ax4.set_ylabel('적절성 점수')
ax4.set_title('효율성 vs 적절성 (색상: 변화율)')
plt.colorbar(scatter, ax=ax4, label='변화율 (%)')
ax4.grid(True, alpha=0.3)
# 5) 현재 vs 최적 진료비
ax5 = axes[1, 1]
ax5.scatter(results_df['현재_진료비(천원)'], results_df['최적_진료비(천원)'], alpha=0.7, color='purple')
ax5.plot([0, results_df['현재_진료비(천원)'].max()], [0, results_df['현재_진료비(천원)'].max()], 'r--', alpha=0.7)
ax5.set_xlabel('현재 진료비 (천원)')
ax5.set_ylabel('최적 진료비 (천원)')
ax5.set_title('현재 vs 최적 진료비')
ax5.grid(True, alpha=0.3)
# 6) 변화율 분포
ax6 = axes[1, 2]
ax6.hist(results_df['변화율(%)'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
ax6.set_xlabel('변화율 (%)')
ax6.set_ylabel('빈도')
ax6.set_title('진료비 변화율 분포')
ax6.axvline(x=0, color='red', linestyle='--', alpha=0.7)
ax6.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/진료비_분배_최적화_시각화_SLSQP.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"✅ 시각화 저장 완료: {output_dir}/진료비_분배_최적화_시각화_SLSQP.png")
print()

print("=== 진료비 분배 최적화 결과 요약 (SLSQP 개선본) ===")
print(f"📊 총 현재 진료비: {총_진료비:,.0f}천원")
print(f"📊 총 최적 진료비: {results_df['최적_진료비(천원)'].sum():,.0f}천원")
print(f"📊 총 변화량: {results_df['변화량(천원)'].sum():,.0f}천원")
print(f"📊 평균 변화율: {results_df['변화율(%)'].mean():.2f}%")
print(f"📊 최적화 성공: {res.success}")
print(f"📊 진료비 추정: {추정_완료}개 성공, {추정_실패}개 실패")
print()
print("🏆 상위 5개 진료과 (변화량 기준):")
top_5 = results_df.groupby('진료과')['변화량(천원)'].sum().sort_values(ascending=False).head()
for i, (dept, change) in enumerate(top_5.items(), 1):
    print(f"  {i}. {dept}: {change:,.0f}천원")
print()
print("🏥 상위 3개 병원 (변화량 기준):")
top_3_hosp = results_df.groupby('병원명')['변화량(천원)'].sum().sort_values(ascending=False).head()
for i, (hosp, change) in enumerate(top_3_hosp.items(), 1):
    print(f"  {i}. {hosp}: {change:,.0f}천원")
print()
print("✅ 최적화 완료! 결과 파일이 저장되었습니다.") 