# run_scenario1.py

import pandas as pd
import numpy as np

# 1. 파일 경로 설정

# 1. 파일 경로 설정 (Windows forward slashes)
PATH_M1 = "C:/Users/jenny/babohun/model_results_연령지역_진료과/Stacking_prediction_results_detailed.csv"
PATH_M2 = "C:/Users/jenny/babohun/analysis_data/병원별_진료과별_입원외래_통합_시계열예측결과_개선.csv"
PATH_COST = "C:/Users/jenny/babohun/src/formodel/진료과-진료비/진료비_구간예측결과.csv"
OUTPUT_DIR = "C:/Users/jenny/babohun"
OUTPUT_DIR = "C:/Users/jenny/babohun"

# 2. 데이터 로드
df_m1   = pd.read_csv(PATH_M1, encoding="utf-8-sig")
df_m2   = pd.read_csv(PATH_M2, encoding="utf-8-sig")
df_cost = pd.read_csv(PATH_COST, encoding="utf-8-sig")

# 3. 모델1·2 예측 통합
m1 = df_m1[['지역','구분','year_num','y_predicted']]
m2 = df_m2[['병원','진료과','연도','RF예측']]
merged = pd.merge(
    m1, m2,
    left_on = ['지역','구분','year_num'],
    right_on= ['병원','진료과','연도'],
    how='inner'
)
merged['predicted_inpatients'] = merged[['y_predicted','RF예측']].mean(axis=1)

# 4. 시나리오1 추천 생성
scenario1 = merged.copy()
scenario1['bed_change_pct']        = 0.0
scenario1['specialist_change_pct'] = 0.0
scenario1['marketing_priority']    = 'Maintain'
scenario1['scenario']              = 1
scenario1.to_csv(f"{OUTPUT_DIR}/scenario1_recommendations.csv", index=False, encoding="utf-8-sig")

# 5. 비용 요약 (모델3)
cost_summary = (
    df_cost
    .groupby('년도')['pred_cost']
    .sum()
    .reset_index()
)
cost_summary['cost_growth'] = cost_summary['pred_cost'].pct_change()
cost_summary['scenario']    = 1
cost_summary.to_csv(f"{OUTPUT_DIR}/scenario1_cost_summary.csv", index=False, encoding="utf-8-sig")

# 6. 주요 지표 계산
national = {
    'inpatient_year_persons': 26992067,
    'inpatient_year_cases':   2979275,
    'outpatient_year_persons':71118807,
    'outpatient_year_cases':  13487671,
    'beds':   109678,
    'doctors':18338,
    'nurses': 187618
}
daily = {
    'inpatient_persons': national['inpatient_year_persons']/365,
    'inpatient_cases':   national['inpatient_year_cases']/365,
    'outpatient_persons':national['outpatient_year_persons']/365,
    'outpatient_cases':  national['outpatient_year_cases']/365
}

metrics = {}
metrics['병상가동률']        = daily['inpatient_persons'] / national['beds'] * 100
metrics['병상당의사수']     = national['doctors'] / national['beds']
metrics['의사당외래환자수'] = daily['outpatient_cases'] / national['doctors']
metrics['외래대입원비율']   = national['inpatient_year_persons'] / national['outpatient_year_persons']

# 7. 평가 리포트 생성 (필요 지표만)
names = ['병상가동률','병상당의사수','의사당외래환자수','외래대입원비율']

def classify(sim, avg):
    lower, upper = avg * 0.95, avg * 1.05
    if sim < lower:
        return '낮음'
    elif sim > upper:
        return '높음'
    else:
        return '적정'

report_rows = []
for name in names:
    sim = metrics[name]
    avg = metrics[name]
    report_rows.append({
        '시나리오':    1,
        '지표명':      name,
        '시뮬레이션값': round(sim, 2),
        '전국평균':    round(avg, 2),
        '평가':        classify(sim, avg)
    })

df_report = pd.DataFrame(report_rows,
    columns=['시나리오','지표명','시뮬레이션값','전국평균','평가']
)
df_report.to_csv(f"{OUTPUT_DIR}/evaluation_report_scenario1.csv", index=False, encoding="utf-8-sig")

print("=== evaluation_report_scenario1 ===")
print(df_report)

scenario1.to_csv(
    "s1r/scenario1_recommendations6.csv",
    index=False,
    encoding="utf-8-sig"
)

# 비용 요약 저장
cost_summary.to_csv(
    "s1r/scenario1_cost_summary6.csv",
    index=False,
    encoding="utf-8-sig"
)

