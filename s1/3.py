import pandas as pd

# 1. 파일 경로 설정 (Windows forward slashes)
PATH_M1 = "C:/Users/jenny/babohun/model_results_연령지역_진료과/Stacking_prediction_results_detailed.csv"
PATH_M2 = "C:/Users/jenny/babohun/analysis_data/병원별_진료과별_입원외래_통합_시계열예측결과_개선.csv"
PATH_COST = "C:/Users/jenny/babohun/src/formodel/진료과-진료비/진료비_구간예측결과.csv"

# 2. 임계값 설정
GROWTH_UPPER = 0.05   # +5% 이상 성장 시 자원 확대 권장
GROWTH_LOWER = -0.05  # -5% 이하 감소 시 자원 축소 권장

# 3. 데이터 로드
# 모델1: ['지역','구분','year_num','y_predicted',...]
df_m1 = pd.read_csv(PATH_M1, encoding='utf-8')
# 모델2: ['병원','진료과','연도','RF예측',...]
df_m2 = pd.read_csv(PATH_M2, encoding='utf-8')
# 모델3: ['년도','pred_cost',...]
df_costs = pd.read_csv(PATH_COST, encoding='utf-8')

# 4. 컬럼명 standard merge keys 확인 및 병합
merged = pd.merge(
    df_m1[['지역','구분','year_num','y_predicted']],
    df_m2[['병원','진료과','연도','RF예측']],
    left_on=['지역','구분','year_num'],
    right_on=['병원','진료과','연도'],
    how='inner'
)
# 환자 수 예측 평균
merged['predicted_inpatients'] = merged[['y_predicted','RF예측']].mean(axis=1)

# 5. 성장률 계산
merged = merged.sort_values(['지역','구분','year_num'])
merged['prev_inpatients'] = merged.groupby(['지역','구분'])['predicted_inpatients'].shift(1)
merged['growth_rate'] = (
    merged['predicted_inpatients'] - merged['prev_inpatients']
) / merged['prev_inpatients']

# 6. 시나리오 ①: 현상 유지 결과 가공
# - 자원 유지: 병상·전문의 변화 없음
# - 일반인 유입: 모니터링 없이 홍보·서비스 개선 제안
# bed_change_pct, specialist_change_pct 0, marketing_priority='Maintain'
scenario1 = merged.copy()
scenario1['bed_change_pct'] = 0
scenario1['specialist_change_pct'] = 0
scenario1['marketing_priority'] = 'Maintain'
scenario1['scenario'] = '현상 유지'

# 7. 경제성 평가 (모델3 비용): '년도' 기준 그룹화
cost_summary = (
    df_costs
    .groupby('년도')['pred_cost']
    .sum()
    .reset_index()
)
cost_summary['cost_growth'] = cost_summary['pred_cost'].pct_change()
cost_summary['scenario'] = '현상 유지'

# 8. 결과 저장 및 출력
scenario1.to_csv("C:/Users/jenny/babohun/scenario1_recommendations2.csv", index=False)
cost_summary.to_csv("C:/Users/jenny/babohun/scenario1_cost_summary2.csv", index=False)

print("=== Scenario1 Recommendations ===")
print(scenario1.head())
print("\n=== Scenario1 Cost Summary ===")
print(cost_summary)
