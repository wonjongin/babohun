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
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from optuna import create_study

# 1. 데이터 불러오기 (더 풍부한 데이터 사용)
print("=== 데이터 로딩 시작 ===")

# 기존 데이터
df_info = pd.read_csv('new_merged_data/병원_통합_데이터.csv')

# 추가 데이터 소스들
try:
    df_pred = pd.read_csv('analysis_data/병원별_진료과별_입원외래_통합_시계열예측결과_개선.csv')
    print("✅ 시계열 예측 데이터 로드 완료")
except:
    print("⚠️ 시계열 예측 데이터 없음, 다른 데이터로 대체")
    df_pred = None

try:
    df_region = pd.read_csv('analysis_data/지역별_의료통계.csv')
    print("✅ 지역별 의료통계 데이터 로드 완료")
except:
    print("⚠️ 지역별 의료통계 데이터 없음")
    df_region = None

try:
    df_department = pd.read_csv('analysis_data/진료과별_통계.csv')
    print("✅ 진료과별 통계 데이터 로드 완료")
except:
    print("⚠️ 진료과별 통계 데이터 없음")
    df_department = None

# 2. 데이터 전처리 및 통합
print("\n=== 데이터 전처리 및 통합 ===")

# 병원 정보 데이터 전처리
df_info['병원명'] = df_info['병원명'].astype(str).str.strip()

# 진료과별 전문의수 컬럼 추출
doc_columns = [col for col in df_info.columns if col.endswith('_전문의수')]
print(f"발견된 전문의수 컬럼: {len(doc_columns)}개")

# 3. 풍부한 피처 생성
print("\n=== 풍부한 피처 생성 ===")

X_rows = []
y_list = []

# 병원별 기본 정보 추출
for idx, row in df_info.iterrows():
    병원명 = row['병원명']
    
    # 각 진료과별로 데이터 생성
    for doc_col in doc_columns:
        진료과 = doc_col.replace('_전문의수', '')
        전문의수 = row[doc_col]
        
        if pd.notnull(전문의수) and 전문의수 > 0:
            # 기본 정보
            row_data = {
                '병원명': 병원명,
                '진료과': 진료과,
                '전문의수': 전문의수
            }
            
            # 1) 병상수 관련 피처 (기존)
            bed_columns = [
                '강내치료실', '격리병실', '무균치료실', '물리치료실', '방사선옥소', '분만실', '수술실', '신생아실', 
                '응급실', '인공신장실', '일반입원실_상급', '일반입원실_일반', '정신과개방_상급', '정신과개방_일반', 
                '정신과폐쇄_상급', '정신과폐쇄_일반', '중환자실_성인', '중환자실_소아', '중환자실_신생아', '회복실',
                '가족실', '간호사실', '목욕실', '상담실', '임종실', '처치실', '화장실'
            ]
            
            for bed_col in bed_columns:
                if bed_col in row.index:
                    bed_val = row[bed_col]
                    row_data[bed_col] = bed_val if pd.notnull(bed_val) else 0
                else:
                    row_data[bed_col] = 0
            
            # 2) 병원 규모 관련 피처
            total_beds = sum([row_data[col] for col in bed_columns if col in row_data])
            row_data['총병상수'] = total_beds
            
            # 병원 규모 분류
            if total_beds >= 1000:
                row_data['병원규모'] = '대형'
            elif total_beds >= 500:
                row_data['병원규모'] = '중형'
            else:
                row_data['병원규모'] = '소형'
            
            # 3) 진료과별 특성 피처
            if '내과' in 진료과:
                row_data['진료과_내과계열'] = 1
            elif '외과' in 진료과 or '정형외과' in 진료과:
                row_data['진료과_외과계열'] = 1
            elif '소아' in 진료과:
                row_data['진료과_소아계열'] = 1
            elif '정신' in 진료과:
                row_data['진료과_정신계열'] = 1
            else:
                row_data['진료과_기타계열'] = 1
            
            # 4) 지역 정보 (병원명에서 추출)
            if '서울' in 병원명:
                row_data['지역'] = '서울'
                row_data['대도시'] = 1
            elif '부산' in 병원명 or '대구' in 병원명 or '인천' in 병원명 or '광주' in 병원명 or '대전' in 병원명:
                row_data['지역'] = '광역시'
                row_data['대도시'] = 1
            else:
                row_data['지역'] = '기타'
                row_data['대도시'] = 0
            
            # 5) 시계열 예측 데이터가 있으면 추가
            if df_pred is not None:
                pred_data = df_pred[(df_pred['병원'] == 병원명) & (df_pred['진료과'] == 진료과)]
                if len(pred_data) > 0:
                    pred_row = pred_data.iloc[0]
                    row_data['ARIMA예측'] = pred_row.get('ARIMA예측', 0)
                    row_data['RF예측'] = pred_row.get('RF예측', 0)
                    row_data['XGB예측'] = pred_row.get('XGB예측', 0)
                    row_data['실제값'] = pred_row.get('실제값', 0)
                else:
                    row_data['ARIMA예측'] = 0
                    row_data['RF예측'] = 0
                    row_data['XGB예측'] = 0
                    row_data['실제값'] = 0
            else:
                row_data['ARIMA예측'] = 0
                row_data['RF예측'] = 0
                row_data['XGB예측'] = 0
                row_data['실제값'] = 0
            
            # 6) 추가 통계 피처
            row_data['병상당전문의수'] = 전문의수 / (total_beds + 1)
            row_data['중환자실비율'] = (row_data.get('중환자실_성인', 0) + row_data.get('중환자실_소아', 0)) / (total_beds + 1)
            row_data['일반입원실비율'] = (row_data.get('일반입원실_상급', 0) + row_data.get('일반입원실_일반', 0)) / (total_beds + 1)
            
            X_rows.append(row_data)
            y_list.append(전문의수)

X = pd.DataFrame(X_rows)
y = pd.Series(y_list, name='전문의수')

print(f"생성된 데이터 크기: {X.shape}")
print(f"타겟 분포:\n{y.describe()}")

# 4. 고급 피처 엔지니어링
print("\n=== 고급 피처 엔지니어링 ===")

# 1) 시계열 예측 관련 피처 (있는 경우)
if 'ARIMA예측' in X.columns and 'RF예측' in X.columns and 'XGB예측' in X.columns:
    # 예측값 통계
    X['예측값_평균'] = X[['ARIMA예측', 'RF예측', 'XGB예측']].mean(axis=1)
    X['예측값_표준편차'] = X[['ARIMA예측', 'RF예측', 'XGB예측']].std(axis=1)
    X['예측값_최대'] = X[['ARIMA예측', 'RF예측', 'XGB예측']].max(axis=1)
    X['예측값_최소'] = X[['ARIMA예측', 'RF예측', 'XGB예측']].min(axis=1)
    
    # 가중 예측값
    X['가중예측값'] = (0.2 * X['ARIMA예측'] + 0.3 * X['RF예측'] + 0.5 * X['XGB예측'])
    
    # 예측 정확도 지표
    X['ARIMA_오차'] = abs(X['ARIMA예측'] - X['실제값'])
    X['RF_오차'] = abs(X['RF예측'] - X['실제값'])
    X['XGB_오차'] = abs(X['XGB예측'] - X['실제값'])
    
    # 로그 변환
    X['ARIMA예측_log'] = np.log1p(np.abs(X['ARIMA예측']))
    X['RF예측_log'] = np.log1p(np.abs(X['RF예측']))
    X['XGB예측_log'] = np.log1p(np.abs(X['XGB예측']))
    X['실제값_log'] = np.log1p(np.abs(X['실제값']))
    
    # 비율 피처
    X['ARIMA_비율'] = np.where(X['실제값'] != 0, X['ARIMA예측'] / X['실제값'], 1.0)
    X['RF_비율'] = np.where(X['실제값'] != 0, X['RF예측'] / X['실제값'], 1.0)
    X['XGB_비율'] = np.where(X['실제값'] != 0, X['XGB예측'] / X['실제값'], 1.0)

# 2) 병상수 관련 고급 피처
X['병상당예측환자수'] = X.get('가중예측값', X['총병상수']) / (X['총병상수'] + 1)

# 병상수 비율들
bed_ratio_columns = [
    '중환자실비율', '일반입원실비율', '병상당전문의수'
]

# 3) 상호작용 피처
X['총병상수_대도시'] = X['총병상수'] * X['대도시']
X['병상당전문의수_대도시'] = X['병상당전문의수'] * X['대도시']

# 4) 다항식 피처
X['총병상수_제곱'] = X['총병상수'] ** 2
X['총병상수_세제곱'] = X['총병상수'] ** 3

# 5) 로그 변환 (큰 값들의 영향 줄이기)
X['총병상수_log'] = np.log1p(X['총병상수'])
X['병상당전문의수_log'] = np.log1p(np.abs(X['병상당전문의수']))

# 6) 범주형 변수 원핫 인코딩
categorical_columns = ['병원명', '진료과', '병원규모', '지역']
X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

print(f"피처 엔지니어링 후 총 피처 수: {X.shape[1]}개")

# 5. 데이터 정제
print("\n=== 데이터 정제 ===")

# NaN 값 처리
X = X.fillna(0)

# 무한대 값 처리
X = X.replace([np.inf, -np.inf], 0)

# 이상치 제거 (99% 분위수)
numeric_columns = X.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    if col != '전문의수':  # 타겟 변수는 제외
        q99 = X[col].quantile(0.99)
        X[col] = np.where(X[col] > q99, q99, X[col])
        X[col] = np.abs(X[col])  # 음수 값 처리

print("데이터 정제 완료")

# 6. 고급 모델링
print("\n=== 고급 모델링 시작 ===")

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=None)

print(f"훈련 데이터 크기: {X_train.shape}")
print(f"테스트 데이터 크기: {X_test.shape}")

# 고급 모델 리스트
models_advanced = {
    'XGBoost': XGBRegressor(
        random_state=42, 
        n_estimators=200, 
        max_depth=6, 
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0
    ),
    'LightGBM': LGBMRegressor(
        verbose=-1, 
        random_state=42, 
        n_estimators=200, 
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0
    ),
    'CatBoost': CatBoostRegressor(
        verbose=0, 
        random_state=42, 
        iterations=200, 
        depth=6,
        learning_rate=0.1,
        l2_leaf_reg=3.0
    ),
    'RandomForest': RandomForestRegressor(
        random_state=42, 
        n_estimators=200, 
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt'
    ),
    'GradientBoosting': GradientBoostingRegressor(
        random_state=42,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8
    ),
    'ElasticNet': ElasticNet(
        random_state=42,
        alpha=0.1,
        l1_ratio=0.5,
        max_iter=2000
    ),
    'SVR': SVR(
        kernel='rbf',
        C=1.0,
        gamma='scale'
    ),
    'KNN': KNeighborsRegressor(
        n_neighbors=5,
        weights='distance'
    )
}

# 교차검증 및 평가
cv = KFold(n_splits=5, shuffle=True, random_state=42)

print("\n=== 모델 성능 평가 ===")
model_results = {}

for name, model in models_advanced.items():
    print(f"\n--- {name} ---")
    
    try:
        # 교차검증
        r2_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
        rmse_scores = np.sqrt(-cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error'))
        mae_scores = -cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_absolute_error')
        
        print(f'  CV R²:   {r2_scores.mean():.4f} ± {r2_scores.std():.4f}')
        print(f'  CV RMSE: {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}')
        print(f'  CV MAE:  {mae_scores.mean():.4f} ± {mae_scores.std():.4f}')
        
        model_results[name] = {
            'r2_mean': r2_scores.mean(),
            'r2_std': r2_scores.std(),
            'rmse_mean': rmse_scores.mean(),
            'rmse_std': rmse_scores.std(),
            'mae_mean': mae_scores.mean(),
            'mae_std': mae_scores.std()
        }
        
    except Exception as e:
        print(f'  오류 발생: {str(e)}')
        continue

# 7. 앙상블 모델
print("\n=== 앙상블 모델 학습 ===")

# 개별 모델 학습
trained_models = {}
for name, model in models_advanced.items():
    try:
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"✅ {name} 학습 완료")
    except Exception as e:
        print(f"❌ {name} 학습 실패: {str(e)}")
        continue

# 성능 기반 가중 앙상블
if len(trained_models) >= 3:
    # R² 기준으로 상위 3개 모델 선택
    top_models = sorted(model_results.items(), key=lambda x: x[1]['r2_mean'], reverse=True)[:3]
    
    ensemble_weights = [0.5, 0.3, 0.2]  # 상위 모델에 더 높은 가중치
    ensemble_models = [(name, trained_models[name]) for name, _ in top_models]
    
    print(f"앙상블 모델: {[name for name, _ in ensemble_models]}")
    print(f"앙상블 가중치: {ensemble_weights}")
    
    # 앙상블 예측
    y_pred_train_ensemble = np.zeros(len(X_train))
    y_pred_test_ensemble = np.zeros(len(X_test))
    
    for (name, model), weight in zip(ensemble_models, ensemble_weights):
        y_pred_train_ensemble += weight * model.predict(X_train)
        y_pred_test_ensemble += weight * model.predict(X_test)
    
    # 앙상블 성능 평가
    ensemble_train_r2 = r2_score(y_train, y_pred_train_ensemble)
    ensemble_test_r2 = r2_score(y_test, y_pred_test_ensemble)
    ensemble_train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train_ensemble))
    ensemble_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test_ensemble))
    
    print(f"앙상블 성능:")
    print(f"  Train R²: {ensemble_train_r2:.4f}")
    print(f"  Test R²:  {ensemble_test_r2:.4f}")
    print(f"  Train RMSE: {ensemble_train_rmse:.4f}")
    print(f"  Test RMSE:  {ensemble_test_rmse:.4f}")

# 8. 결과 저장
print("\n=== 결과 저장 ===")

# 결과 저장 디렉토리 생성
import os
results_dir = "model_results_진료과_전문의_연령지역진료과"
os.makedirs(f"{results_dir}/performance", exist_ok=True)
os.makedirs(f"{results_dir}/predictions", exist_ok=True)
os.makedirs(f"{results_dir}/models", exist_ok=True)

# 성능 비교 결과 저장
if model_results:
    performance_df = pd.DataFrame(model_results).T
    performance_df = performance_df.sort_values('r2_mean', ascending=False)
    
    performance_df.to_csv(f"{results_dir}/performance/model_performance_comparison.csv", encoding='utf-8-sig')
    
    print("\n📊 모델 성능 비교 (CV R² 기준 내림차순):")
    print(performance_df[['r2_mean', 'r2_std', 'rmse_mean', 'mae_mean']].round(4))
    
    print(f"✅ 성능 비교 결과 저장: {results_dir}/performance/model_performance_comparison.csv")

# 개별 모델 예측 결과 저장
for name, model in trained_models.items():
    try:
        y_pred = model.predict(X_test)
        
        pred_df = X_test.copy()
        pred_df['y_actual'] = y_test.values
        pred_df['y_predicted'] = y_pred
        pred_df['prediction_error'] = y_test.values - y_pred
        pred_df['absolute_error'] = np.abs(y_test.values - y_pred)
        pred_df['model'] = name
        
        pred_df.to_csv(f"{results_dir}/predictions/{name}_predictions.csv", encoding='utf-8-sig', index=False)
        print(f"✅ {name} 예측 결과 저장: {results_dir}/predictions/{name}_predictions.csv")
        
    except Exception as e:
        print(f"❌ {name} 예측 결과 저장 실패: {str(e)}")

# 앙상블 예측 결과 저장
if 'y_pred_test_ensemble' in locals():
    ensemble_pred_df = X_test.copy()
    ensemble_pred_df['y_actual'] = y_test.values
    ensemble_pred_df['y_predicted'] = y_pred_test_ensemble
    ensemble_pred_df['prediction_error'] = y_test.values - y_pred_test_ensemble
    ensemble_pred_df['absolute_error'] = np.abs(y_test.values - y_pred_test_ensemble)
    ensemble_pred_df['model'] = 'Weighted_Ensemble'
    
    ensemble_pred_df.to_csv(f"{results_dir}/predictions/Weighted_Ensemble_predictions.csv", encoding='utf-8-sig', index=False)
    print(f"✅ 앙상블 예측 결과 저장: {results_dir}/predictions/Weighted_Ensemble_predictions.csv")

# 모델 저장
import joblib
for name, model in trained_models.items():
    try:
        model_path = f"{results_dir}/models/{name}_model.pkl"
        joblib.dump(model, model_path)
        print(f"✅ {name} 모델 저장: {model_path}")
    except Exception as e:
        print(f"❌ {name} 모델 저장 실패: {str(e)}")

print("\n" + "="*60)
print("🎉 개선된 모델링 완료!")
print("="*60)
print(f"📁 결과 파일 위치: {results_dir}/")
print("📊 성능 비교: performance/model_performance_comparison.csv")
print("🎯 예측 결과: predictions/")
print("💾 모델 파일: models/")

# 최종 성능 요약
if model_results:
    best_model_name = performance_df.index[0]
    best_performance = performance_df.loc[best_model_name]
    
    print(f"\n🏆 최고 성능 모델: {best_model_name}")
    print(f"   - CV R²: {best_performance['r2_mean']:.4f} ± {best_performance['r2_std']:.4f}")
    print(f"   - CV RMSE: {best_performance['rmse_mean']:.4f} ± {best_performance['rmse_std']:.4f}")
    print(f"   - CV MAE: {best_performance['mae_mean']:.4f} ± {best_performance['mae_std']:.4f}")

if 'ensemble_test_r2' in locals():
    print(f"\n🎯 앙상블 모델 성능:")
    print(f"   - Test R²: {ensemble_test_r2:.4f}")
    print(f"   - Test RMSE: {ensemble_test_rmse:.4f}")

print("="*60)

'''
=== 데이터 로딩 시작 ===
✅ 시계열 예측 데이터 로드 완료
⚠️ 지역별 의료통계 데이터 없음
⚠️ 진료과별 통계 데이터 없음

=== 데이터 전처리 및 통합 ===
발견된 전문의수 컬럼: 34개

=== 풍부한 피처 생성 ===
생성된 데이터 크기: (128, 46)
타겟 분포:
count    128.000000
mean       3.156250
std        4.563528
min        1.000000
25%        1.000000
50%        2.000000
75%        3.000000
max       47.000000
Name: 전문의수, dtype: float64

=== 고급 피처 엔지니어링 ===
피처 엔지니어링 후 총 피처 수: 101개

=== 데이터 정제 ===
데이터 정제 완료

=== 고급 모델링 시작 ===
훈련 데이터 크기: (102, 101)
테스트 데이터 크기: (26, 101)

=== 모델 성능 평가 ===

--- XGBoost ---
  CV R²:   0.8616 ± 0.2143
  CV RMSE: 1.8098 ± 2.8042
  CV MAE:  0.4610 ± 0.6353

--- LightGBM ---
  CV R²:   -0.3756 ± 1.0564
  CV RMSE: 3.5547 ± 2.6899
  CV MAE:  1.6389 ± 0.5950

--- CatBoost ---
  CV R²:   0.6127 ± 0.3041
  CV RMSE: 2.6713 ± 3.0004
  CV MAE:  0.8298 ± 0.6282

--- RandomForest ---
  CV R²:   0.7025 ± 0.2536
  CV RMSE: 2.4020 ± 3.1107
  CV MAE:  0.9474 ± 0.7542

--- GradientBoosting ---
  CV R²:   0.6112 ± 0.4940
  CV RMSE: 2.4201 ± 2.9134
  CV MAE:  0.5814 ± 0.6543

--- ElasticNet ---
  CV R²:   0.9998 ± 0.0003
  CV RMSE: 0.0594 ± 0.0982
  CV MAE:  0.0226 ± 0.0318

--- SVR ---
  CV R²:   0.0377 ± 0.1340
  CV RMSE: 3.7754 ± 3.1633
  CV MAE:  1.7404 ± 0.8313

--- KNN ---
  CV R²:   0.1181 ± 0.1259
  CV RMSE: 3.5521 ± 2.8582
  CV MAE:  1.7266 ± 0.7524

=== 앙상블 모델 학습 ===
✅ XGBoost 학습 완료
✅ LightGBM 학습 완료
✅ CatBoost 학습 완료
✅ RandomForest 학습 완료
✅ GradientBoosting 학습 완료
✅ ElasticNet 학습 완료
✅ SVR 학습 완료
✅ KNN 학습 완료
앙상블 모델: ['ElasticNet', 'XGBoost', 'RandomForest']
앙상블 가중치: [0.5, 0.3, 0.2]
앙상블 성능:
  Train R²: 0.9832
  Test R²:  0.9863
  Train RMSE: 0.6373
  Test RMSE:  0.3041

=== 결과 저장 ===

📊 모델 성능 비교 (CV R² 기준 내림차순):
                  r2_mean  r2_std  rmse_mean  mae_mean
ElasticNet         0.9998  0.0003     0.0594    0.0226
XGBoost            0.8616  0.2143     1.8098    0.4610
RandomForest       0.7025  0.2536     2.4020    0.9474
CatBoost           0.6127  0.3041     2.6713    0.8298
GradientBoosting   0.6112  0.4940     2.4201    0.5814
KNN                0.1181  0.1259     3.5521    1.7266
SVR                0.0377  0.1340     3.7754    1.7404
LightGBM          -0.3756  1.0564     3.5547    1.6389
✅ 성능 비교 결과 저장: model_results_진료과_전문의_연령지역진료과/performance/model_performance_comparison.csv
✅ XGBoost 예측 결과 저장: model_results_진료과_전문의_연령지역진료과/predictions/XGBoost_predictions.csv
✅ LightGBM 예측 결과 저장: model_results_진료과_전문의_연령지역진료과/predictions/LightGBM_predictions.csv
✅ CatBoost 예측 결과 저장: model_results_진료과_전문의_연령지역진료과/predictions/CatBoost_predictions.csv
✅ RandomForest 예측 결과 저장: model_results_진료과_전문의_연령지역진료과/predictions/RandomForest_predictions.csv
✅ GradientBoosting 예측 결과 저장: model_results_진료과_전문의_연령지역진료과/predictions/GradientBoosting_predictions.csv
✅ ElasticNet 예측 결과 저장: model_results_진료과_전문의_연령지역진료과/predictions/ElasticNet_predictions.csv
✅ SVR 예측 결과 저장: model_results_진료과_전문의_연령지역진료과/predictions/SVR_predictions.csv
✅ KNN 예측 결과 저장: model_results_진료과_전문의_연령지역진료과/predictions/KNN_predictions.csv
✅ 앙상블 예측 결과 저장: model_results_진료과_전문의_연령지역진료과/predictions/Weighted_Ensemble_predictions.csv
✅ XGBoost 모델 저장: model_results_진료과_전문의_연령지역진료과/models/XGBoost_model.pkl
✅ LightGBM 모델 저장: model_results_진료과_전문의_연령지역진료과/models/LightGBM_model.pkl
✅ CatBoost 모델 저장: model_results_진료과_전문의_연령지역진료과/models/CatBoost_model.pkl
✅ RandomForest 모델 저장: model_results_진료과_전문의_연령지역진료과/models/RandomForest_model.pkl
✅ GradientBoosting 모델 저장: model_results_진료과_전문의_연령지역진료과/models/GradientBoosting_model.pkl
✅ ElasticNet 모델 저장: model_results_진료과_전문의_연령지역진료과/models/ElasticNet_model.pkl
✅ SVR 모델 저장: model_results_진료과_전문의_연령지역진료과/models/SVR_model.pkl
✅ KNN 모델 저장: model_results_진료과_전문의_연령지역진료과/models/KNN_model.pkl

============================================================
🎉 개선된 모델링 완료!
============================================================
📁 결과 파일 위치: model_results_진료과_전문의_연령지역진료과/
📊 성능 비교: performance/model_performance_comparison.csv
🎯 예측 결과: predictions/
💾 모델 파일: models/

🏆 최고 성능 모델: ElasticNet
   - CV R²: 0.9998 ± 0.0003
   - CV RMSE: 0.0594 ± 0.0982
   - CV MAE: 0.0226 ± 0.0318

🎯 앙상블 모델 성능:
   - Test R²: 0.9863
   - Test RMSE: 0.3041
============================================================

'''
