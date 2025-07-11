import pandas as pd

# 데이터 로드
df = pd.read_csv('analysis_data/병원별_진료과별_입원외래_통합_시계열예측결과_개선.csv')

# 올바른 전체 성능 계산
def calculate_overall_metrics(df, actual_col, pred_col):
    rmse = ((df[actual_col] - df[pred_col])**2).mean()**0.5
    mae = abs(df[actual_col] - df[pred_col]).mean()
    mape = calculate_mape(df[actual_col], df[pred_col])
    return rmse, mae, mape

def calculate_mape(actual, predicted):
    # 실제값이 0이 아닌 경우만 계산
    mask = actual != 0
    if mask.sum() == 0:
        return float('inf')
    return (abs(actual[mask] - predicted[mask]) / actual[mask] * 100).mean()

# 2023년 데이터로 계산
df_2023 = df[df['연도'] == 2023]

arima_rmse, arima_mae, arima_mape = calculate_overall_metrics(df_2023, '실제값', 'ARIMA예측')
rf_rmse, rf_mae, rf_mape = calculate_overall_metrics(df_2023, '실제값', 'RF예측')
xgb_rmse, xgb_mae, xgb_mape = calculate_overall_metrics(df_2023, '실제값', 'XGB예측')

# 전체 평균 계산
summary_stats = {
    'ARIMA': {
        '평균_RMSE': arima_rmse,
        '평균_MAE': arima_mae,
        '평균_MAPE': arima_mape,
        '평균_R2': df_2023['ARIMA_R2'].mean()
    },
    'RF': {
        '평균_RMSE': rf_rmse,
        '평균_MAE': rf_mae,
        '평균_MAPE': rf_mape,
        '평균_R2': df_2023['RF_R2'].mean()
    },
    'XGB': {
        '평균_RMSE': xgb_rmse,
        '평균_MAE': xgb_mae,
        '평균_MAPE': xgb_mape,
        '평균_R2': df_2023['XGB_R2'].mean()
    }
}

# 표로 정리
summary_df = pd.DataFrame(summary_stats).T
print("=== 전체 평균 성능 지표 ===")
print(summary_df.round(2))