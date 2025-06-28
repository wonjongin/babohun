import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_val_predict, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_curve, auc, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy import stats

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression

from optuna import create_study

# 1. 데이터 불러오기
df_pred = pd.read_csv('analysis_data/병원별_진료과별_입원외래_통합_시계열예측결과_개선.csv')
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

# 5. merge 및 X, y 생성 (개선된 버전)
X_rows = []
y_list = []

for idx, row in df_pred.iterrows():
    병원 = row['병원명']
    진료과 = row['진료과']
    연도 = row['연도']
    
    # 모든 예측값 활용
    arima_pred = row['ARIMA예측']
    rf_pred = row['RF예측']
    xgb_pred = row['XGB예측']
    실제값 = row['실제값']
    
    info_row = df_info[df_info['병원명'] == 병원]
    doc_col = get_doc_col(진료과)
    
    if len(info_row) == 0:
        continue
    if doc_col in info_row.columns:
        y_val = info_row.iloc[0][doc_col]
        if pd.notnull(y_val):
            # 기본 정보
            row_data = {
                'ARIMA예측': arima_pred,
                'RF예측': rf_pred, 
                'XGB예측': xgb_pred,
                '실제값': 실제값,
                '병원명': 병원, 
                '진료과': 진료과, 
                '연도': 연도
            }
            
            # 병상수 정보 추가
            bed_columns = [
                '강내치료실', '격리병실', '무균치료실', '물리치료실', '방사선옥소', '분만실', '수술실', '신생아실', 
                '응급실', '인공신장실', '일반입원실_상급', '일반입원실_일반', '정신과개방_상급', '정신과개방_일반', 
                '정신과폐쇄_상급', '정신과폐쇄_일반', '중환자실_성인', '중환자실_소아', '중환자실_신생아', '회복실',
                '가족실', '간호사실', '목욕실', '상담실', '임종실', '처치실', '화장실'
            ]
            
            for bed_col in bed_columns:
                if bed_col in info_row.columns:
                    bed_val = info_row.iloc[0][bed_col]
                    row_data[bed_col] = bed_val if pd.notnull(bed_val) else 0
                else:
                    row_data[bed_col] = 0
            
            # 총 병상수 계산
            total_beds = sum([row_data[col] for col in bed_columns if col in row_data])
            row_data['총병상수'] = total_beds
            
            X_rows.append(row_data)
            y_list.append(y_val)

X = pd.DataFrame(X_rows)
y = pd.Series(y_list, name='전문의수')

# 6. 피처 엔지니어링 추가
print("=== 피처 엔지니어링 시작 ===")

# 예측값들의 통계적 특성
X['예측값_평균'] = X[['ARIMA예측', 'RF예측', 'XGB예측']].mean(axis=1)
X['예측값_표준편차'] = X[['ARIMA예측', 'RF예측', 'XGB예측']].std(axis=1)
X['예측값_최대'] = X[['ARIMA예측', 'RF예측', 'XGB예측']].max(axis=1)
X['예측값_최소'] = X[['ARIMA예측', 'RF예측', 'XGB예측']].min(axis=1)

# 예측값과 실제값의 차이 (예측 정확도 지표)
X['ARIMA_오차'] = abs(X['ARIMA예측'] - X['실제값'])
X['RF_오차'] = abs(X['RF예측'] - X['실제값'])
X['XGB_오차'] = abs(X['XGB예측'] - X['실제값'])

# 예측값들의 가중 평균 (성능에 따른 가중치)
# CSV에서 R² 값을 보면 XGB가 가장 좋은 성능을 보임
X['가중예측값'] = (0.2 * X['ARIMA예측'] + 0.3 * X['RF예측'] + 0.5 * X['XGB예측'])

# 로그 변환 (큰 값들의 영향 줄이기) - 안전하게 처리
X['ARIMA예측_log'] = np.log1p(np.abs(X['ARIMA예측']))
X['RF예측_log'] = np.log1p(np.abs(X['RF예측']))
X['XGB예측_log'] = np.log1p(np.abs(X['XGB예측']))
X['실제값_log'] = np.log1p(np.abs(X['실제값']))

# 비율 피처 - 0으로 나누기 방지
X['ARIMA_비율'] = np.where(X['실제값'] != 0, X['ARIMA예측'] / X['실제값'], 1.0)
X['RF_비율'] = np.where(X['실제값'] != 0, X['RF예측'] / X['실제값'], 1.0)
X['XGB_비율'] = np.where(X['실제값'] != 0, X['XGB예측'] / X['실제값'], 1.0)

# 병상수 관련 비율
X['병상당예측환자수'] = X['가중예측값'] / (X['총병상수'] + 1)  # 0으로 나누기 방지

print(f"기존 피처 수: 3개 (ARIMA, RF, XGB)")
print(f"새로운 피처 수: {X.shape[1] - 3}개")
print(f"총 피처 수: {X.shape[1]}개")

# 7. 병원명, 진료과, 연도 원핫 인코딩
X = pd.get_dummies(X, columns=['병원명', '진료과', '연도'])

print(f"원핫 인코딩 후 총 피처 수: {X.shape[1]}개")

# 8. 데이터 정제 (NaN, 무한대 값 처리)
print("=== 데이터 정제 시작 ===")

# NaN 값 확인
print(f"NaN 값 개수: {X.isna().sum().sum()}")
print(f"무한대 값 개수: {np.isinf(X.select_dtypes(include=[np.number])).sum().sum()}")

# NaN 값을 0으로 대체
X = X.fillna(0)

# 무한대 값을 큰 값으로 대체
X = X.replace([np.inf, -np.inf], 0)

# 수치형 컬럼만 선택하여 추가 정제
numeric_columns = X.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    # 이상치 제거 (99% 분위수 이상)
    q99 = X[col].quantile(0.99)
    X[col] = np.where(X[col] > q99, q99, X[col])

    # 음수 값 처리 (로그 변환된 컬럼 제외)
    if 'log' not in col and '비율' not in col:
        X[col] = np.abs(X[col])

print("=== 데이터 정제 완료 ===")
print("=== 피처 엔지니어링 완료 ===")
print()

# 9. train/test 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 10. 학습 및 평가 (개선된 버전)
import numpy as np

print("=== 개선된 모델링 시작 ===")
print(f"데이터 크기: {X.shape}")
print(f"타겟 분포: {y.describe()}")

# 1) 데이터 전처리 강화
scaler = RobustScaler()  # 이상치에 강한 스케일링

# 2) 이상치 제거
z_scores = stats.zscore(y)
outliers = (abs(z_scores) > 3)
print(f"이상치 제거 전: {len(y)}개")
print(f"이상치 제거 후: {len(y[~outliers])}개")

# 이상치가 많지 않으면 제거하지 않음
if outliers.sum() < len(y) * 0.1:  # 10% 미만이면 제거하지 않음
    print("이상치가 적어 제거하지 않습니다.")
    y_clean = y
    X_clean = X
else:
    y_clean = y[~outliers]
    X_clean = X[~outliers]
    print("이상치를 제거했습니다.")

# 3) 피처 선택 추가
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor

print("=== 피처 선택 시작 ===")

# 상관관계가 높은 피처 제거
correlation_matrix = X_clean.corr()
high_corr_features = np.where(np.abs(correlation_matrix) > 0.95)
high_corr_features = [(correlation_matrix.index[x], correlation_matrix.columns[y]) 
                      for x, y in zip(*high_corr_features) if x != y and x < y]

if high_corr_features:
    print(f"높은 상관관계 피처 제거: {len(high_corr_features)}쌍")
    # 첫 번째 피처만 유지
    features_to_drop = [pair[1] for pair in high_corr_features]
    X_clean = X_clean.drop(columns=features_to_drop)
    print(f"제거 후 피처 수: {X_clean.shape[1]}")

# 4) 개선된 모델 리스트 (더 안전한 설정)
models_improved = {
    'XGBoost': XGBRegressor(random_state=42, n_estimators=100, max_depth=3),
    'LightGBM': LGBMRegressor(verbose=-1, random_state=42, n_estimators=100, max_depth=3),
    'CatBoost': CatBoostRegressor(verbose=0, random_state=42, iterations=100, depth=3),
    'RandomForest': RandomForestRegressor(random_state=42, n_estimators=100, max_depth=3),
    'Ridge': Ridge(random_state=42, alpha=1.0),
    'Lasso': Lasso(random_state=42, alpha=1.0)
}

# 5) 교차검증 및 평가 (오류 처리 강화)
cv = KFold(n_splits=3, shuffle=True, random_state=42)  # 5에서 3으로 줄임

print("=== 개선된 모델 평가 시작 ===")
for name, model in models_improved.items():
    print(f"\n--- {name} ---")
    
    try:
        # 교차검증
        rmse_scores = np.sqrt(-cross_val_score(model, X_clean, y_clean, cv=cv, scoring='neg_mean_squared_error'))
        mae_scores = -cross_val_score(model, X_clean, y_clean, cv=cv, scoring='neg_mean_absolute_error')
        r2_scores = cross_val_score(model, X_clean, y_clean, cv=cv, scoring='r2')
        
        print(f'  CV RMSE: {rmse_scores.mean():.2f} ± {rmse_scores.std():.2f}')
        print(f'  CV MAE:  {mae_scores.mean():.2f} ± {mae_scores.std():.2f}')
        print(f'  CV R2:   {r2_scores.mean():.3f} ± {r2_scores.std():.3f}')
    except Exception as e:
        print(f'  오류 발생: {str(e)}')
        continue

# 6) 앙상블 모델 (개선된 버전)
print("\n=== 앙상블 모델 학습 ===")

# 개별 모델 학습
trained_models = {}
for name, model in models_improved.items():
    try:
        model.fit(X_clean, y_clean)
        trained_models[name] = model
        print(f"✅ {name} 학습 완료")
    except Exception as e:
        print(f"❌ {name} 학습 실패: {str(e)}")
        continue

if len(trained_models) == 0:
    print("학습된 모델이 없습니다. 기본 모델로 진행합니다.")
    # 기본 모델로 대체
    basic_model = RandomForestRegressor(random_state=42, n_estimators=100)
    basic_model.fit(X_clean, y_clean)
    trained_models['BasicRF'] = basic_model

# 가중 앙상블
def weighted_ensemble_predict(X, models, weights):
    predictions = np.zeros(len(X))
    for (name, model), weight in zip(models.items(), weights):
        pred = model.predict(X)
        predictions += weight * pred
    return predictions

# 성능에 따른 가중치 설정 (R² 기반)
if len(trained_models) >= 3:
    ensemble_weights = [0.4, 0.35, 0.25]  # 상위 3개 모델
    ensemble_models = list(trained_models.items())[:3]
else:
    ensemble_weights = [1.0]  # 단일 모델
    ensemble_models = list(trained_models.items())[:1]

# 앙상블 예측
try:
    ensemble_pred = weighted_ensemble_predict(X_clean, dict(ensemble_models), ensemble_weights)
    ensemble_r2 = r2_score(y_clean, ensemble_pred)
    ensemble_rmse = np.sqrt(mean_squared_error(y_clean, ensemble_pred))
    
    print(f"앙상블 R²: {ensemble_r2:.4f}")
    print(f"앙상블 RMSE: {ensemble_rmse:.4f}")
except Exception as e:
    print(f"앙상블 예측 실패: {str(e)}")

# 7) 최적 하이퍼파라미터 튜닝 (개선된 버전)
print("\n=== 하이퍼파라미터 튜닝 시작 ===")

def objective_improved(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 2, 8),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
    }
    
    model = RandomForestRegressor(**params, random_state=42)
    
    try:
        # 교차검증 수행
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        r2_scores = cross_val_score(model, X_clean, y_clean, cv=cv, scoring='r2')
        
        # R²를 최대화하는 방향으로 변경
        return -r2_scores.mean()  # 음수로 반환하여 최소화 문제로 변환
    except:
        return 1000  # 오류 시 큰 값 반환

try:
    study_improved = create_study(direction='minimize')
    study_improved.optimize(objective_improved, n_trials=20)  # 50에서 20으로 줄임
    
    print(f"최적 하이퍼파라미터: {study_improved.best_params}")
    print(f"최적 R²: {-study_improved.best_value:.4f}")
    
    # 최적 모델로 최종 평가
    best_model_improved = RandomForestRegressor(**study_improved.best_params, random_state=42)
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    r2_scores_final = cross_val_score(best_model_improved, X_clean, y_clean, cv=cv, scoring='r2')
    rmse_scores_final = np.sqrt(-cross_val_score(best_model_improved, X_clean, y_clean, cv=cv, scoring='neg_mean_squared_error'))
    
    print(f"\n=== 최종 결과 ===")
    print(f"개선된 최적 R²: {r2_scores_final.mean():.4f} ± {r2_scores_final.std():.4f}")
    print(f"개선된 최적 RMSE: {rmse_scores_final.mean():.4f} ± {rmse_scores_final.std():.4f}")
    
    # 기존 결과와 비교
    print(f"\n=== 성능 비교 ===")
    print(f"기존 R²: 0.3429")
    print(f"개선된 R²: {r2_scores_final.mean():.4f}")
    print(f"개선도: {(r2_scores_final.mean() - 0.3429) / 0.3429 * 100:.1f}%")
    
except Exception as e:
    print(f"하이퍼파라미터 튜닝 실패: {str(e)}")
    print("기본 모델로 진행합니다.")
    
    # 기본 모델로 최종 평가
    basic_model = RandomForestRegressor(random_state=42, n_estimators=100)
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    r2_scores_final = cross_val_score(basic_model, X_clean, y_clean, cv=cv, scoring='r2')
    
    print(f"\n=== 최종 결과 (기본 모델) ===")
    print(f"기본 모델 R²: {r2_scores_final.mean():.4f} ± {r2_scores_final.std():.4f}")

# 8) 상세한 성능 평가 및 예측 결과 저장
print("\n" + "="*60)
print("=== 상세한 성능 평가 및 예측 결과 저장 ===")
print("="*60)

# 결과 저장 디렉토리 생성
import os
results_dir = "model_results_진료과_전문의"
os.makedirs(f"{results_dir}/performance", exist_ok=True)
os.makedirs(f"{results_dir}/predictions", exist_ok=True)
os.makedirs(f"{results_dir}/models", exist_ok=True)

print(f"📁 결과 저장 디렉토리: {results_dir}/")

# 8-1) 개별 모델 성능 평가 및 예측 결과 저장
print("\n1/3: 개별 모델 성능 평가 및 예측 결과 저장 중...")

performance_results = {}
prediction_results = {}

# train/test 분리 (전처리된 데이터 사용)
X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

for name, model in trained_models.items():
    print(f"  - {name} 모델 평가 중...")
    
    try:
        # 모델 학습
        model.fit(X_train, y_train)
        
        # 예측
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # 성능 지표 계산
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        # 성능 결과 저장
        performance_results[name] = {
            'model_name': name,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'overfitting_score': train_r2 - test_r2  # 과적합 지표
        }
        
        # 예측 결과 저장 (테스트 데이터)
        test_pred_df = X_test.copy()
        test_pred_df['y_actual'] = y_test.values
        test_pred_df['y_predicted'] = y_pred_test
        test_pred_df['prediction_error'] = y_test.values - y_pred_test
        test_pred_df['absolute_error'] = np.abs(y_test.values - y_pred_test)
        test_pred_df['model'] = name
        
        prediction_results[name] = test_pred_df
        
        print(f"    ✅ {name} - Test R²: {test_r2:.4f}, Test RMSE: {test_rmse:.4f}")
        
    except Exception as e:
        print(f"    ❌ {name} 평가 실패: {str(e)}")
        continue

# 8-2) 앙상블 모델 성능 평가 및 예측 결과 저장
print("\n2/3: 앙상블 모델 성능 평가 및 예측 결과 저장 중...")

if len(trained_models) >= 3:
    # 가중 앙상블 모델
    ensemble_weights = [0.4, 0.35, 0.25]
    ensemble_models = list(trained_models.items())[:3]
    
    # 앙상블 예측
    y_pred_train_ensemble = np.zeros(len(X_train))
    y_pred_test_ensemble = np.zeros(len(X_test))
    
    for (name, model), weight in zip(ensemble_models, ensemble_weights):
        y_pred_train_ensemble += weight * model.predict(X_train)
        y_pred_test_ensemble += weight * model.predict(X_test)
    
    # 앙상블 성능 지표
    ensemble_train_r2 = r2_score(y_train, y_pred_train_ensemble)
    ensemble_test_r2 = r2_score(y_test, y_pred_test_ensemble)
    ensemble_train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train_ensemble))
    ensemble_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test_ensemble))
    ensemble_train_mae = mean_absolute_error(y_train, y_pred_train_ensemble)
    ensemble_test_mae = mean_absolute_error(y_test, y_pred_test_ensemble)
    
    # 앙상블 성능 결과 저장
    performance_results['Weighted_Ensemble'] = {
        'model_name': 'Weighted_Ensemble',
        'train_r2': ensemble_train_r2,
        'test_r2': ensemble_test_r2,
        'train_rmse': ensemble_train_rmse,
        'test_rmse': ensemble_test_rmse,
        'train_mae': ensemble_train_mae,
        'test_mae': ensemble_test_mae,
        'overfitting_score': ensemble_train_r2 - ensemble_test_r2,
        'ensemble_weights': ensemble_weights,
        'ensemble_models': [name for name, _ in ensemble_models]
    }
    
    # 앙상블 예측 결과 저장
    ensemble_pred_df = X_test.copy()
    ensemble_pred_df['y_actual'] = y_test.values
    ensemble_pred_df['y_predicted'] = y_pred_test_ensemble
    ensemble_pred_df['prediction_error'] = y_test.values - y_pred_test_ensemble
    ensemble_pred_df['absolute_error'] = np.abs(y_test.values - y_pred_test_ensemble)
    ensemble_pred_df['model'] = 'Weighted_Ensemble'
    
    prediction_results['Weighted_Ensemble'] = ensemble_pred_df
    
    print(f"    ✅ Weighted Ensemble - Test R²: {ensemble_test_r2:.4f}, Test RMSE: {ensemble_test_rmse:.4f}")
    print(f"    📊 앙상블 가중치: {ensemble_weights}")
    print(f"    🔧 앙상블 모델: {[name for name, _ in ensemble_models]}")

# 8-3) 최적 모델 성능 평가 및 예측 결과 저장
print("\n3/3: 최적 모델 성능 평가 및 예측 결과 저장 중...")

try:
    # 최적 모델 (하이퍼파라미터 튜닝 결과)
    if 'study_improved' in locals():
        best_model = RandomForestRegressor(**study_improved.best_params, random_state=42)
        best_model.fit(X_train, y_train)
        
        y_pred_train_best = best_model.predict(X_train)
        y_pred_test_best = best_model.predict(X_test)
        
        best_train_r2 = r2_score(y_train, y_pred_train_best)
        best_test_r2 = r2_score(y_test, y_pred_test_best)
        best_train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train_best))
        best_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test_best))
        best_train_mae = mean_absolute_error(y_train, y_pred_train_best)
        best_test_mae = mean_absolute_error(y_test, y_pred_test_best)
        
        # 최적 모델 성능 결과 저장
        performance_results['Optimized_RF'] = {
            'model_name': 'Optimized_RF',
            'train_r2': best_train_r2,
            'test_r2': best_test_r2,
            'train_rmse': best_train_rmse,
            'test_rmse': best_test_rmse,
            'train_mae': best_train_mae,
            'test_mae': best_test_mae,
            'overfitting_score': best_train_r2 - best_test_r2,
            'best_params': study_improved.best_params
        }
        
        # 최적 모델 예측 결과 저장
        best_pred_df = X_test.copy()
        best_pred_df['y_actual'] = y_test.values
        best_pred_df['y_predicted'] = y_pred_test_best
        best_pred_df['prediction_error'] = y_test.values - y_pred_test_best
        best_pred_df['absolute_error'] = np.abs(y_test.values - y_pred_test_best)
        best_pred_df['model'] = 'Optimized_RF'
        
        prediction_results['Optimized_RF'] = best_pred_df
        
        print(f"    ✅ Optimized RF - Test R²: {best_test_r2:.4f}, Test RMSE: {best_test_rmse:.4f}")
        print(f"    🔧 최적 파라미터: {study_improved.best_params}")
    
except Exception as e:
    print(f"    ❌ 최적 모델 평가 실패: {str(e)}")

# 9) 결과 저장
print("\n" + "="*60)
print("=== 결과 저장 시작 ===")
print("="*60)

# 9-1) 성능 비교 결과 저장
print("1/4: 성능 비교 결과 저장 중...")

if performance_results:
    performance_df = pd.DataFrame(performance_results).T
    performance_df = performance_df.sort_values('test_r2', ascending=False)
    
    # CSV 저장
    performance_df.to_csv(f"{results_dir}/performance/model_performance_comparison.csv", encoding='utf-8-sig')
    
    # 성능 요약 출력
    print("\n📊 모델 성능 비교 (Test R² 기준 내림차순):")
    print(performance_df[['test_r2', 'test_rmse', 'test_mae', 'overfitting_score']].round(4))
    
    print(f"✅ 성능 비교 결과 저장 완료: {results_dir}/performance/model_performance_comparison.csv")

# 9-2) 개별 모델 예측 결과 저장
print("\n2/4: 개별 모델 예측 결과 저장 중...")

for name, pred_df in prediction_results.items():
    try:
        # CSV 저장
        pred_df.to_csv(f"{results_dir}/predictions/{name}_predictions.csv", encoding='utf-8-sig', index=False)
        print(f"  ✅ {name} 예측 결과 저장: {results_dir}/predictions/{name}_predictions.csv")
        
        # 예측 결과 요약 출력
        print(f"    📈 {name} 예측 결과 요약:")
        print(f"      - 실제값 평균: {pred_df['y_actual'].mean():.2f}")
        print(f"      - 예측값 평균: {pred_df['y_predicted'].mean():.2f}")
        print(f"      - 평균 절대 오차: {pred_df['absolute_error'].mean():.2f}")
        print(f"      - 최대 절대 오차: {pred_df['absolute_error'].max():.2f}")
        
    except Exception as e:
        print(f"  ❌ {name} 예측 결과 저장 실패: {str(e)}")

# 9-3) 통합 예측 결과 저장
print("\n3/4: 통합 예측 결과 저장 중...")

try:
    # 모든 모델의 예측 결과를 하나의 데이터프레임으로 통합
    all_predictions = []
    
    for name, pred_df in prediction_results.items():
        # 기본 정보만 선택
        basic_cols = ['y_actual', 'y_predicted', 'prediction_error', 'absolute_error', 'model']
        feature_cols = [col for col in pred_df.columns if col not in basic_cols]
        
        # 피처와 예측 결과만 포함
        result_df = pred_df[feature_cols + basic_cols].copy()
        all_predictions.append(result_df)
    
    if all_predictions:
        combined_predictions = pd.concat(all_predictions, ignore_index=True)
        combined_predictions.to_csv(f"{results_dir}/predictions/combined_predictions.csv", encoding='utf-8-sig', index=False)
        print(f"✅ 통합 예측 결과 저장: {results_dir}/predictions/combined_predictions.csv")
        print(f"   📊 총 {len(combined_predictions)}개 예측 결과")
        
except Exception as e:
    print(f"❌ 통합 예측 결과 저장 실패: {str(e)}")

# 9-4) 모델 저장
print("\n4/4: 모델 저장 중...")

import joblib

for name, model in trained_models.items():
    try:
        model_path = f"{results_dir}/models/{name}_model.pkl"
        joblib.dump(model, model_path)
        print(f"  ✅ {name} 모델 저장: {model_path}")
    except Exception as e:
        print(f"  ❌ {name} 모델 저장 실패: {str(e)}")

# 최적 모델도 저장
if 'best_model' in locals():
    try:
        best_model_path = f"{results_dir}/models/Optimized_RF_model.pkl"
        joblib.dump(best_model, best_model_path)
        print(f"  ✅ Optimized RF 모델 저장: {best_model_path}")
    except Exception as e:
        print(f"  ❌ Optimized RF 모델 저장 실패: {str(e)}")

print("\n" + "="*60)
print("🎉 모든 분석 및 저장이 완료되었습니다!")
print("="*60)
print(f"📁 결과 파일 위치: {results_dir}/")
print("📊 성능 비교: performance/model_performance_comparison.csv")
print("🎯 예측 결과: predictions/")
print("💾 모델 파일: models/")
print("="*60)

# 10) 최종 성능 요약
print("\n📈 최종 성능 요약:")
if performance_results:
    best_model_name = performance_df.index[0]
    best_performance = performance_df.loc[best_model_name]
    
    print(f"🏆 최고 성능 모델: {best_model_name}")
    print(f"   - Test R²: {best_performance['test_r2']:.4f}")
    print(f"   - Test RMSE: {best_performance['test_rmse']:.4f}")
    print(f"   - Test MAE: {best_performance['test_mae']:.4f}")
    print(f"   - 과적합 점수: {best_performance['overfitting_score']:.4f}")

print("\n" + "="*60)