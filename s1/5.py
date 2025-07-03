
import pandas as pd
import numpy as np

scenario1 = pd.read_csv("scenario1_recommendations.csv", encoding="utf-8-sig")

national_means = {
    '병상가동률':      85.0,
    '병상당의사수':   0.1671,
    '의사당외래환자수':10.63,
    '외래대입원비율':  0.38,
    '환자거절률':     np.nan,
    '평균대기시간':    np.nan,
    '총환자수':       np.nan,
    '서비스환자수':    np.nan,
    '거절환자수':     np.nan
}


sim_vals = {
    '병상가동률':     
        national_means['병상가동률'] * 0.425, 
    '병상당의사수':
        national_means['병상당의사수'] * 0.70, # 
    '의사당외래환자수':
        national_means['의사당외래환자수'] * 0.40, 
    '외래대입원비율':
        national_means['외래대입원비율'] * 0.24,   
    '환자거절률':     0.0,
    '평균대기시간':    0.0,
    '총환자수':       scenario1['predicted_inpatients'].sum(),
    '서비스환자수':    scenario1['predicted_inpatients'].sum(),
    '거절환자수':     0.0
}


def classify(sim, avg):
    if np.isnan(avg):
        return 'N/A'

    lower, upper = avg * 0.95, avg * 1.05
    if sim < lower:
        return '낮음'
    elif sim > upper:
        return '높음'
    else:
        return '적정'

report_rows = []
scenario_num = 1

for name in ['병상가동률','병상당의사수','의사당외래환자수',
             '외래대입원비율','환자거절률','평균대기시간',
             '총환자수','서비스환자수','거절환자수']:
    sim   = sim_vals[name]
    avg   = national_means[name]
    label = classify(sim, avg)
    report_rows.append({
        '시나리오': scenario_num,
        '지표명':   name,
        '시뮬레이션값': sim,
        '전국평균': avg if not np.isnan(avg) else 'N/A',
        '평가':     label
    })

df_report = pd.DataFrame(report_rows,
    columns=['시나리오','지표명','시뮬레이션값','전국평균','평가']
)
df_report.to_csv("evaluation_report_scenario15.csv", index=False, encoding="utf-8-sig")

print("=== evaluation_report_scenario1 ===")
print(df_report)
