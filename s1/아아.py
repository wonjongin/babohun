import pandas as pd

# 1. 파일 경로 설정 (Windows raw strings)
PATH_M1 = r"C:/Users/jenny/babohun/model_results_연령지역_진료과/Stacking_prediction_results_detailed.csv"
PATH_M2 = r"C:/Users/jenny/babohun/analysis_data/병원별_진료과별_입원외래_통합_시계열예측결과_개선.csv"
PATH_COST = r"C:/Users/jenny/babohun/src/formodel/진료과-진료비/진료비_구간예측결과.csv"

# 2. 임계값 설정
GROWTH_UPPER = 0.05  # +5% 이상 성장 시 자원 확대 권장
GROWTH_LOWER = -0.05 # -5% 이하 감소 시 자원 축소 권장

# 3. 데이터 로드
df_m1 = pd.read_csv(PATH_M1, encoding='utf-8')  # 모델1 예측치 로드
df_m2 = pd.read_csv(PATH_M2, encoding='utf-8')  # 모델2 예측치 로드
df_costs = pd.read_csv(PATH_COST, encoding='utf-8')  # 모델3 예측 진료비 로드

# 4. 컬럼명 확인 및 필요 시 rename
print("Model1 columns:", df_m1.columns.tolist())
print("Model2 columns:", df_m2.columns.tolist())
# 예: df_m1.rename(columns={'old_region':'region', ...}, inplace=True)
#     df_m2.rename(columns={'old_department':'department', ...}, inplace=True)

# 5. 모델1·2 예측 통합
# 실제 컬럼명이 'predicted_patient_count'이 아닐 경우, 위 rename 이후 맞춰서 사용하세요
df_pred = (
    df_m1.merge(df_m2, on=['region','department','year'], suffixes=('_m1','_m2'))
)
df_pred['predicted_inpatients'] = df_pred[['predicted_patient_count_m1','predicted_patient_count_m2']].mean(axis=1)

# 6. 성장률 계산
df_pred = df_pred.sort_values(['region','department','year'])
df_pred['prev_inpatients'] = df_pred.groupby(['region','department'])['predicted_inpatients'].shift(1)
df_pred['growth_rate'] = (df_pred['predicted_inpatients'] - df_pred['prev_inpatients']) / df_pred['prev_inpatients']

# 7. 재배치 전략 제안
def adjust_factor(rate):
    if rate >= GROWTH_UPPER:
        return 1.15  # +15%
    elif rate <= GROWTH_LOWER:
        return 0.90  # -10%
    return 1.00      # 유지

df_pred['adjustment_factor'] = df_pred['growth_rate'].apply(adjust_factor)
df_pred['bed_change_pct'] = (df_pred['adjustment_factor'] - 1) * 100
df_pred['specialist_change_pct'] = df_pred['bed_change_pct']

# 8. 일반인 유입 전략 제안
def marketing_priority(rate):
    if rate < GROWTH_LOWER:
        return 'High'
    elif rate < 0:
        return 'Medium'
    return 'Low'

df_pred['marketing_priority'] = df_pred['growth_rate'].apply(marketing_priority)

# 9. 경제성 평가
cost_summary = df_costs.groupby('year')['predicted_cost'].sum().reset_index()
cost_summary['cost_growth'] = cost_summary['predicted_cost'].pct_change()

# 10. 결과 저장 및 출력
df_pred.to_csv(r"C:/Users/jenny/babohun/scenario1_recommendations.csv", index=False)
cost_summary.to_csv(r"C:/Users/jenny/babohun/scenario1_cost_summary.csv", index=False)

print("=== Recommendations (First 5 rows) ===")
print(df_pred.head())
print("\n=== Cost Summary ===")
print(cost_summary)
