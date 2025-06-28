import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 데이터 불러오기
df = pd.read_csv('new_merged_data/연도별진료과별_입원외래_통합환자수.csv')
df['전체환자수_합계'] = pd.to_numeric(df['전체환자수_합계'], errors='coerce').fillna(0)

def calculate_metrics(y_true, y_pred, n_features=1):
    """개선된 평가 지표 계산"""
    # 기본 지표
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # MAPE - 0이 아닌 값만 계산
    nonzero_mask = y_true > 0
    if np.any(nonzero_mask):
        mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100
    else:
        mape = np.nan
    
    # R²
    r2 = r2_score(y_true, y_pred)
    
    # Adjusted R² - 분모가 0보다 클 때만 계산
    n = len(y_true)
    if n - n_features - 1 > 0:
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    else:
        adjusted_r2 = np.nan
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2,
        'Adjusted_R2': adjusted_r2
    }

# 연도 범위 확인 및 설정
available_years = sorted(df['연도'].unique())
print(f"사용 가능한 연도: {available_years}")

# 최소 3개 연도 이상 데이터가 있는 조합만 분석
min_years = 3
results = []
model_performance = {
    'ARIMA': {'RMSE': [], 'MAE': [], 'MAPE': [], 'R2': [], 'Adjusted_R2': []},
    'RF': {'RMSE': [], 'MAE': [], 'MAPE': [], 'R2': [], 'Adjusted_R2': []},
    'XGB': {'RMSE': [], 'MAE': [], 'MAPE': [], 'R2': [], 'Adjusted_R2': []}
}

total_combinations = 0
valid_combinations = 0

for 병원 in df['병원'].unique():
    for 진료과 in df['진료과'].unique():
        total_combinations += 1
        
        # 해당 병원×진료과의 데이터 추출
        subset = df[(df['병원'] == 병원) & (df['진료과'] == 진료과)].sort_values('연도')
        
        # 최소 연도 수 확인
        if len(subset) < min_years:
            continue
            
        valid_combinations += 1
        print(f"분석 중: {병원} {진료과} ({len(subset)}개 연도)")
        
        # 시계열 데이터 준비
        ts_data = subset.set_index('연도')['전체환자수_합계']

        # 최소 2개 이상의 0이 아닌 값이 있는지 확인
        if (ts_data > 0).sum() < 2:
            continue

        # 훈련/테스트 분할 (마지막 1개를 테스트)
        if len(ts_data) >= 3:
            train = ts_data.iloc[:-1]
            test = ts_data.iloc[-1:]
        else:
            continue

        # ARIMA 모델
        try:
            # 데이터가 너무 적으면 간단한 모델 사용
            if len(train) < 5:
                arima_model = ARIMA(train, order=(0,0,0)).fit()
            else:
            arima_model = ARIMA(train, order=(1,1,1)).fit()
            
            arima_pred = arima_model.forecast(steps=len(test))
            arima_metrics = calculate_metrics(test.values, arima_pred)
            
            # 모델 성능 저장
            for metric, value in arima_metrics.items():
                if not np.isnan(value):
                    model_performance['ARIMA'][metric].append(value)
                
        except Exception as e:
            print(f"  ARIMA 오류: {e}")
            arima_pred = np.array([np.nan] * len(test))
            arima_metrics = {metric: np.nan for metric in ['RMSE', 'MAE', 'MAPE', 'R2', 'Adjusted_R2']}

        # Random Forest & XGBoost
        try:
            # 연도를 feature로 사용
            X_train = np.array(train.index).reshape(-1,1).astype(int)
            y_train = train.values
            X_test = np.array(test.index).reshape(-1,1).astype(int)
            y_test = test.values
            
            # 데이터 검증
            print(f"    훈련 데이터: {len(X_train)}개, 테스트 데이터: {len(X_test)}개")
            print(f"    훈련 값: {y_train}")
            print(f"    테스트 값: {y_test}")

            # RF 모델
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(X_train, y_train)
            rf_pred = rf.predict(X_test)
            rf_metrics = calculate_metrics(y_test, rf_pred)

            # XGB 모델
            xgb = XGBRegressor(n_estimators=50, random_state=42)
            xgb.fit(X_train, y_train)
            xgb_pred = xgb.predict(X_test)
            xgb_metrics = calculate_metrics(y_test, xgb_pred)

            # 모델 성능 저장
            for metric in ['RMSE', 'MAE', 'MAPE', 'R2', 'Adjusted_R2']:
                if not np.isnan(rf_metrics[metric]):
                    model_performance['RF'][metric].append(rf_metrics[metric])
                if not np.isnan(xgb_metrics[metric]):
                    model_performance['XGB'][metric].append(xgb_metrics[metric])

            # 결과 저장
        for i, year in enumerate(test.index):
                # ARIMA 예측값 처리 (numpy array이므로 인덱스로 접근)
                arima_pred_value = arima_pred[i] if isinstance(arima_pred, np.ndarray) else arima_pred.iloc[i] if hasattr(arima_pred, 'iloc') else arima_pred
                
            results.append({
                '병원': 병원,
                '진료과': 진료과,
                '연도': year,
                    '실제값': y_test[i],
                    'ARIMA예측': arima_pred_value if not np.isnan(arima_pred_value) else '',
                'RF예측': rf_pred[i],
                'XGB예측': xgb_pred[i],
                    'ARIMA_RMSE': arima_metrics['RMSE'],
                    'ARIMA_MAE': arima_metrics['MAE'],
                    'ARIMA_MAPE': arima_metrics['MAPE'],
                    'ARIMA_R2': arima_metrics['R2'],
                    'ARIMA_Adjusted_R2': arima_metrics['Adjusted_R2'],
                    'RF_RMSE': rf_metrics['RMSE'],
                    'RF_MAE': rf_metrics['MAE'],
                    'RF_MAPE': rf_metrics['MAPE'],
                    'RF_R2': rf_metrics['R2'],
                    'RF_Adjusted_R2': rf_metrics['Adjusted_R2'],
                    'XGB_RMSE': xgb_metrics['RMSE'],
                    'XGB_MAE': xgb_metrics['MAE'],
                    'XGB_MAPE': xgb_metrics['MAPE'],
                    'XGB_R2': xgb_metrics['R2'],
                    'XGB_Adjusted_R2': xgb_metrics['Adjusted_R2']
                })
                
        except Exception as e:
            print(f"  RF/XGB 오류: {e}")
            continue

print(f"\n총 조합 수: {total_combinations}")
print(f"유효한 조합 수: {valid_combinations}")

# DataFrame으로 변환 및 저장
if results:
result_df = pd.DataFrame(results)
    result_df.to_csv('analysis_data/병원별_진료과별_입원외래_통합_시계열예측결과_개선.csv', index=False, encoding='utf-8')
    print('저장 완료: analysis_data/병원별_진료과별_입원외래_통합_시계열예측결과_개선.csv')

    # 모델별 전체 평균 성능 계산 및 출력
    print("\n=== 모델별 전체 평균 성능 ===")
    for model_name, metrics in model_performance.items():
        print(f"\n{model_name} 모델:")
        for metric_name, values in metrics.items():
            if values:  # 빈 리스트가 아닌 경우만
                avg_value = np.mean(values)
                print(f"  {metric_name}: {avg_value:.4f} (n={len(values)})")
            else:
                print(f"  {metric_name}: 계산 불가")

    # 모델별 성능 비교
    print("\n=== 모델별 성능 순위 ===")
    for metric in ['RMSE', 'MAE', 'MAPE', 'R2', 'Adjusted_R2']:
        print(f"\n{metric} 기준:")
        model_scores = {}
        for model_name, metrics in model_performance.items():
            if metrics[metric]:  # 빈 리스트가 아닌 경우만
                model_scores[model_name] = np.mean(metrics[metric])
        
        if model_scores:
            # RMSE, MAE, MAPE는 낮을수록 좋음, R²는 높을수록 좋음
            if metric in ['RMSE', 'MAE', 'MAPE']:
                sorted_models = sorted(model_scores.items(), key=lambda x: x[1])
            else:
                sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            
            for i, (model, score) in enumerate(sorted_models, 1):
                print(f"  {i}위: {model} ({score:.4f})")

    print(f"\n총 데이터 포인트: {len(result_df)}")
else:
    print("분석 가능한 데이터가 없습니다.")

# 데이터 분포 확인
print("\n=== 데이터 분포 ===")
print(df.groupby(['병원', '진료과'])['전체환자수_합계'].sum().reset_index()) 

'''
저장 완료: analysis_data/병원별_진료과별_입원외래_통합_시계열예측결과_개선.csv
ARIMA 전체 평균 RMSE: 6767.05
RF 전체 평균 RMSE: 5747.34
XGB 전체 평균 RMSE: 5324.78
'''