import pandas as pd

# 1. 파일 경로 설정 (Windows forward slashes)
PATH_M1 = "C:/Users/jenny/babohun/model_results_연령지역_진료과/Stacking_prediction_results_detailed.csv"
PATH_M2 = "C:/Users/jenny/babohun/analysis_data/병원별_진료과별_입원외래_통합_시계열예측결과_개선.csv"
PATH_COST = "C:/Users/jenny/babohun/src/formodel/진료과-진료비/진료비_구간예측결과.csv"

# 2. 임계값 설정
GROWTH_UPPER = 0.05   # +5% 이상 성장 시 자원 확대 권장
GROWTH_LOWER = -0.05  # -5% 이하 감소 시 자원 축소 권장

# 3. 데이터 로드
df_m1 = pd.read_csv(PATH_M1, encoding='utf-8')   # 모델1 예측치 로드
# ['year_num','age_num',...,'지역','구분',...,'y_predicted',...]
df_m2 = pd.read_csv(PATH_M2, encoding='utf-8')   # 모델2 예측치 로드
# ['병원','진료과','연도','실제값','RF예측',...]
df_costs = pd.read_csv(PATH_COST, encoding='utf-8')  # 모델3 진료비 예측 로드
# ['년도','지역','구분','진료비',...,'pred_cost']

# 4. 모델1·2 예측 통합
# 지역(병원), 구분(진료과), 연도(year_num vs 연도)를 기준으로 병합
merged = pd.merge(
    df_m1[['지역','구분','year_num','y_predicted']],
    df_m2[['병원','진료과','연도','RF예측']],
    left_on=['지역','구분','year_num'],
    right_on=['병원','진료과','연도'],
    how='inner'
)
# 예측 환자 수 평균 계산
merged['predicted_inpatients'] = merged[['y_predicted','RF예측']].mean(axis=1)

# 5. 성장률 계산
merged = merged.sort_values(['지역','구분','year_num'])
merged['prev_inpatients'] = merged.groupby(['지역','구분'])['predicted_inpatients'].shift(1)
merged['growth_rate'] = (
    merged['predicted_inpatients'] - merged['prev_inpatients']
) / merged['prev_inpatients']

# 6. 재배치 전략 제안

def adjust_factor(rate):
    if rate >= GROWTH_UPPER:
        return 1.15
    elif rate <= GROWTH_LOWER:
        return 0.90
    return 1.00

merged['adjustment_factor'] = merged['growth_rate'].apply(adjust_factor)
merged['bed_change_pct'] = (merged['adjustment_factor'] - 1) * 100
merged['specialist_change_pct'] = merged['bed_change_pct']

# 7. 일반인 유입 전략 제안

def marketing_priority(rate):
    if rate < GROWTH_LOWER:
        return 'High'
    elif rate < 0:
        return 'Medium'
    return 'Low'

merged['marketing_priority'] = merged['growth_rate'].apply(marketing_priority)

# 8. 경제성 평가
# '년도' 기준으로 pred_cost 합산
cost_summary = (
    df_costs
    .groupby('년도')['pred_cost']
    .sum()
    .reset_index()
)
cost_summary['cost_growth'] = cost_summary['pred_cost'].pct_change()

# 9. 결과 저장 및 출력
merged.to_csv("C:/Users/jenny/babohun/scenario1_recommendations.csv", index=False)
cost_summary.to_csv("C:/Users/jenny/babohun/scenario1_cost_summary.csv", index=False)

print("=== Recommendations (First 5 rows) ===")
print(merged.head())
print("\n=== Cost Summary ===")
print(cost_summary)
