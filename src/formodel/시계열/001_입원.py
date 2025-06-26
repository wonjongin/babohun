import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# 데이터 불러오기
df = pd.read_csv('new_merged_data/연도별진료과별입원환자수.csv')
df['전체환자수'] = pd.to_numeric(df['전체환자수'], errors='coerce').fillna(0)

years = list(map(str, range(2013, 2024)))
results = []

for 병원 in df['병원'].unique():
    for 진료과 in df['진료과'].unique():
        ts = df[(df['병원'] == 병원) & (df['진료과'] == 진료과)].sort_values('연도')
        ts = ts.set_index('연도')['전체환자수']
        ts.index = ts.index.astype(str)
        ts = ts.reindex(years, fill_value=0)
        train = ts.iloc[:-2]
        test = ts.iloc[-2:]

        # 데이터가 모두 0이면 스킵
        if train.sum() == 0 and test.sum() == 0:
            continue

        # ARIMA
        try:
            arima_model = ARIMA(train, order=(1,1,1)).fit()
            arima_pred = arima_model.forecast(steps=2)
        except:
            arima_pred = np.array([np.nan, np.nan])

        # 2. SARIMA (계절성: 연간, 1년 주기)
        # sarima_model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,1)).fit(disp=False)
        # sarima_pred = sarima_model.forecast(steps=2)
        # print('SARIMA 예측:', sarima_pred.values)

        # 3. Random Forest 회귀 (연도를 feature로 사용)
        X_train = np.array(train.index).reshape(-1,1).astype(int)
        y_train = train.values
        X_test = np.array(test.index).reshape(-1,1).astype(int)

        rf = RandomForestRegressor()
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)

        # 4. XGBoost 회귀
        xgb = XGBRegressor()
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict(X_test)

        # RMSE
        rmse_arima = np.sqrt(mean_squared_error(test.values, arima_pred))
        # print('SARIMA RMSE:', mean_squared_error(test.values, sarima_pred.values, squared=False))
        rmse_rf = np.sqrt(mean_squared_error(test.values, rf_pred))
        rmse_xgb = np.sqrt(mean_squared_error(test.values, xgb_pred))

        # 결과 저장 (연도별로)
        for i, year in enumerate(test.index):
            results.append({
                '병원': 병원,
                '진료과': 진료과,
                '연도': year,
                '실제값': test.values[i],
                'ARIMA예측': arima_pred[i] if not np.isnan(arima_pred[i]) else '',
                'RF예측': rf_pred[i],
                'XGB예측': xgb_pred[i],
                'ARIMA_RMSE': rmse_arima,
                'RF_RMSE': rmse_rf,
                'XGB_RMSE': rmse_xgb
            })

# DataFrame으로 변환 및 저장
result_df = pd.DataFrame(results)
result_df.to_csv('analysis_data/병원별_진료과별_시계열예측결과.csv', index=False, encoding='utf-8')
print('저장 완료: analysis_data/병원별_진료과별_시계열예측결과.csv')

# 모델별 전체 평균 RMSE 계산 및 출력
mean_arima_rmse = result_df['ARIMA_RMSE'].mean()
mean_rf_rmse = result_df['RF_RMSE'].mean()
mean_xgb_rmse = result_df['XGB_RMSE'].mean()
print(f'ARIMA 전체 평균 RMSE: {mean_arima_rmse:.2f}')
print(f'RF 전체 평균 RMSE: {mean_rf_rmse:.2f}')
print(f'XGB 전체 평균 RMSE: {mean_xgb_rmse:.2f}')

print(df['병원'].unique())
print(df['진료과'].unique())

print(df[(df['병원'] == '대구') & (df['진료과'] == '신경외과')])

print(df.groupby(['병원', '진료과'])['전체환자수'].sum().reset_index())


'''
저장 완료: analysis_data/병원별_진료과별_시계열예측결과.csv
ARIMA 전체 평균 RMSE: 3308.31
RF 전체 평균 RMSE: 2537.03
XGB 전체 평균 RMSE: 2313.21
'''