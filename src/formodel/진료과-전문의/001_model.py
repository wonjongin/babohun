import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# 1. 데이터 불러오기
df_pred = pd.read_csv('analysis_data/병원별_진료과별_입원외래_통합_시계열예측결과.csv')
df_info = pd.read_csv('new_merged_data/병원_통합_데이터.csv')

# 2. 최근 연도(2023)만 사용
df_pred = df_pred[df_pred['연도'] == 2023]

# 3. 병원명 컬럼명 통일 및 전처리
df_pred['병원명'] = df_pred['병원'].astype(str).str.strip()
df_pred['진료과'] = df_pred['진료과'].astype(str).str.strip()
df_pred['연도'] = df_pred['연도'].astype(str).str.strip()
df_info['병원명'] = df_info['병원명'].astype(str).str.strip()

# 4. 진료과별 전문의수 컬럼명 추출
def get_doc_col(진료과):
    return f"{진료과}_전문의수"

# 5. merge 및 X, y 생성
X_rows = []
y_list = []
for idx, row in df_pred.iterrows():
    병원 = row['병원명']
    진료과 = row['진료과']
    연도 = row['연도']
    xgb_pred = row['XGB예측']
    info_row = df_info[df_info['병원명'] == 병원]
    doc_col = get_doc_col(진료과)
    if len(info_row) == 0:
        continue
    if doc_col in info_row.columns:
        y_val = info_row.iloc[0][doc_col]
        if pd.notnull(y_val):
            X_rows.append({'XGB예측': xgb_pred, '병원명': 병원, '진료과': 진료과, '연도': 연도})
            y_list.append(y_val)

X = pd.DataFrame(X_rows)
y = pd.Series(y_list, name='전문의수')

# 6. 병원명, 진료과, 연도 원핫 인코딩
X = pd.get_dummies(X, columns=['병원명', '진료과', '연도'])

# 7. train/test 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. 모델 리스트
models = {
    'XGBoost': XGBRegressor(),
    'LightGBM': LGBMRegressor(),
    'CatBoost': CatBoostRegressor(verbose=0),
    'RandomForest': RandomForestRegressor(),
    'LinearRegression': LinearRegression()
}

# 9. 학습 및 평가
import numpy as np
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    print(f'{name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.3f}')

'''
XGBoost: RMSE=1.59, MAE=1.25, R2=-0.181
LightGBM: RMSE=1.32, MAE=1.02, R2=0.183
CatBoost: RMSE=1.48, MAE=1.14, R2=-0.023
RandomForest: RMSE=1.56, MAE=1.15, R2=-0.133
LinearRegression: RMSE=1.32, MAE=1.06, R2=0.192


XGBoost: RMSE=1.31, MAE=1.02, R2=0.203
LightGBM: RMSE=1.32, MAE=1.02, R2=0.183
CatBoost: RMSE=1.15, MAE=0.86, R2=0.384
RandomForest: RMSE=1.11, MAE=0.80, R2=0.423
LinearRegression: RMSE=1.32, MAE=1.07, R2=0.186
'''