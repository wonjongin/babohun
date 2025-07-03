import pandas as pd

# 1. 파일 경로 설정 (Windows forward slashes)
PATH_M1 = "C:/Users/jenny/babohun/model_results_연령지역_진료과/Stacking_prediction_results_detailed.csv"
PATH_M2 = "C:/Users/jenny/babohun/analysis_data/병원별_진료과별_입원외래_통합_시계열예측결과_개선.csv"
PATH_COST = "C:/Users/jenny/babohun/src/formodel/진료과-진료비/진료비_구간예측결과.csv"

# 2. 임계값 설정
GROWTH_UPPER = 0.05   # +5% 이상 성장 시 자원 확대 권장
GROWTH_LOWER = -0.05  # -5% 이하 감소 시 자원 축소 권장

# 3. 데이터 로드
# 모델1, 모델2 예측치 로드
df_m1 = pd.read_csv(PATH_M1, encoding='utf-8')  # ['지역','구분','year_num','y_predicted',...]
df_m2 = pd.read_csv(PATH_M2, encoding='utf-8')  # ['병원','진료과','연도','RF예측',...]
# 모델3 진료비 예측 로드
df_costs = pd.read_csv(PATH_COST, encoding='utf-8')  # ['년도','pred_cost',...]

# 4. 모델1·2 예측 통합
m1 = df_m1[['지역','구분','year_num','y_predicted']]
m2 = df_m2[['병원','진료과','연도','RF예측']]
merged = pd.merge(
    m1, m2,
    left_on=['지역','구분','year_num'],
    right_on=['병원','진료과','연도'],
    how='inner'
)
merged['predicted_inpatients'] = merged[['y_predicted','RF예측']].mean(axis=1)

# 5. 성장률 계산
merged = merged.sort_values(['지역','구분','year_num'])
merged['prev_inpatients'] = merged.groupby(['지역','구분'])['predicted_inpatients'].shift(1)
merged['growth_rate'] = (
    merged['predicted_inpatients'] - merged['prev_inpatients']
) / merged['prev_inpatients']

# 6. 시나리오① 가공: 현상 유지
scenario1 = merged.copy()
scenario1['bed_change_pct'] = 0
scenario1['specialist_change_pct'] = 0
scenario1['marketing_priority'] = 'Maintain'
scenario1['scenario'] = '현상 유지'

# 7. 경제성 평가 (모델3 비용)
cost_summary = (
    df_costs
    .groupby('년도')['pred_cost']
    .sum()
    .reset_index()
)
cost_summary['cost_growth'] = cost_summary['pred_cost'].pct_change()
cost_summary['scenario'] = '현상 유지'

# 8. 보훈병원 시나리오 평가 지표 계산
# 주요 지표의 국가 기준값 사용
national = {
    'inpatient_year_persons': 26992067,
    'inpatient_year_cases': 2979275,
    'outpatient_year_persons': 71118807,
    'outpatient_year_cases': 13487671,
    'beds': 109678,
    'doctors': 18338,
    'nurses': 187618
}
# 일평균 환자 수
daily = {
    'inpatient_persons': national['inpatient_year_persons'] / 365,
    'inpatient_cases': national['inpatient_year_cases'] / 365,
    'outpatient_persons': national['outpatient_year_persons'] / 365,
    'outpatient_cases': national['outpatient_year_cases'] / 365
}
# 1. 병상가동률
metrics = {}
metrics['병상가동률(연인원기준)'] = daily['inpatient_persons'] / national['beds'] * 100
metrics['병상가동률(실인원기준)'] = daily['inpatient_cases'] / national['beds'] * 100
# 2. 병상당 의사·간호사 수
metrics['병상당 의사수'] = national['doctors'] / national['beds']
metrics['병상당 간호사수'] = national['nurses'] / national['beds']
# 3. 의사당 입원환자수
metrics['의사당 입원환자수(연인원)'] = daily['inpatient_persons'] / national['doctors']
metrics['의사당 입원환자수(실인원)'] = daily['inpatient_cases'] / national['doctors']
# 4. 의사당 외래환자수
metrics['의사당 외래환자수(연인원)'] = daily['outpatient_persons'] / national['doctors']
metrics['의사당 외래환자수(실인원)'] = daily['outpatient_cases'] / national['doctors']
# 5. 입원환자당 간호사수
metrics['입원환자당 간호사수(연인원)'] = national['nurses'] / daily['inpatient_persons']
metrics['입원환자당 간호사수(실인원)'] = national['nurses'] / daily['inpatient_cases']
# 6. 외래환자 대비 입원환자비
metrics['외래대입원비'] = national['inpatient_year_persons'] / national['outpatient_year_persons']
# 7. Costliness Index (CI)
metrics['CI'] = 1.0
# 경제성 지표: 연도별 cost_growth 평균
metrics['평균진료비증감률'] = cost_summary['cost_growth'].mean()

# DataFrame 생성 및 저장
df_metrics = pd.DataFrame.from_dict(metrics, orient='index', columns=['value'])
df_metrics.index.name = 'indicator'
df_metrics.to_csv("C:/Users/jenny/babohun/scenario1_indicators.csv")

# 9. 결과 저장 및 출력
scenario1.to_csv("C:/Users/jenny/babohun/scenario1_recommendations4.csv", index=False)
cost_summary.to_csv("C:/Users/jenny/babohun/scenario1_cost_summary4.csv", index=False)

print("=== Scenario1 Recommendations (첫 5개) ===")
print(scenario1.head())
print("\n=== Scenario1 Cost Summary ===")
print(cost_summary)
print("\n=== Scenario1 Evaluation Indicators ===")
print(df_metrics)
