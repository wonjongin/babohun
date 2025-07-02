import pandas as pd
import os

files = [
    ('모델1', 'model_results_연령지역_진료과/Stacking_prediction_results_detailed.csv'),
    ('모델3-연령지역', 'model_results_v3_연령지역진료과_확장/prediction_results_2.csv'),
    ('모델4-연령지역', 'model_results_진료과별병상수_예측모델_연령지역진료과추가/hospital_bed_prediction_results_Ridge_gridcv.csv'),
    ('모델5-연령지역', 'model_results_진료과_전문의_연령지역진료과/predictions/ElasticNet_predictions.csv'),
    ('모델3-시계열', 'model_results_v3_시계열_확장/prediction_results_2.csv'),
    ('모델4-시계열', 'model_results_진료과별병상수_예측모델_시계열추가_3개년/hospital_bed_prediction_results_Ridge_gridcv.csv'),
    ('모델5-시계열', 'model_results_진료과_전문의/predictions/Ridge_predictions.csv'),
]

for name, path in files:
    print(f'\n===== {name} ({path}) =====')
    if not os.path.exists(path):
        print('파일이 존재하지 않습니다.')
        continue
    try:
        df = pd.read_csv(path)
        print(f'컬럼명: {list(df.columns)}')
        print('상위 5행:')
        print(df.head())
    except Exception as e:
        print(f'불러오기 오류: {e}')
