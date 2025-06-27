import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_val_predict, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_curve, auc, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_regression
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
    'LightGBM': LGBMRegressor(verbose=-1),
    'CatBoost': CatBoostRegressor(verbose=0),
    'RandomForest': RandomForestRegressor(),
    'LinearRegression': LinearRegression()
}

# 9. 학습 및 평가
import numpy as np

# 1) 일관된 평가 지표 사용
# 회귀 문제라면: RMSE, MAE, R²만 사용
# 분류 문제라면: Accuracy, Precision, Recall, ROC AUC만 사용

# 2) 데이터 전처리 강화
scaler = RobustScaler()  # 이상치에 강한 스케일링

# 3) 이상치 제거
z_scores = stats.zscore(y)
outliers = (abs(z_scores) > 3)
y_clean = y[~outliers]

# 1) 교차검증 방식 개선
tscv = TimeSeriesSplit(n_splits=5)

# 2) 하이퍼파라미터 튜닝
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None]
}
grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=tscv)

# 3) 앙상블 모델
ensemble = VotingRegressor([
    ('rf', RandomForestRegressor()),
    ('xgb', XGBRegressor()),
    ('cat', CatBoostRegressor(verbose=False))
])

cv = KFold(n_splits=5, shuffle=True, random_state=42)
adj_r2_scores = []

for name, model in models.items():
    # neg_mean_squared_error는 음수로 반환되므로, -를 붙여서 양수로 변환
    rmse_scores = np.sqrt(-cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error'))
    mae_scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    print(f'[{name}]')
    print(f'  CV RMSE: {rmse_scores.mean():.2f} ± {rmse_scores.std():.2f}')
    print(f'  CV MAE:  {mae_scores.mean():.2f} ± {mae_scores.std():.2f}')
    print(f'  CV R2:   {r2_scores.mean():.3f} ± {r2_scores.std():.3f}')

    # 별도 평가용 fit
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    # 테스트셋 평가
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    n = X_test.shape[0]
    p = X_test.shape[1]
    if n > p + 1:
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    else:
        adj_r2 = np.nan  # 계산 불가
    print(f'  Test RMSE: {rmse:.2f}')
    print(f'  Test MAE:  {mae:.2f}')
    print(f'  Test R2:   {r2:.3f}')
    print(f'  Test Adjusted R2: {adj_r2:.3f}')

    # Cross Validation에서 Adjusted R² 계산
    for train_idx, test_idx in cv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        r2 = r2_score(y_te, y_pred)
        n = X_te.shape[0]
        p = X_te.shape[1]
        if n > p + 1:
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        else:
            adj_r2 = np.nan  # 계산 불가
        adj_r2_scores.append(adj_r2)

    print(f'CV Adjusted R2: {np.nanmean(adj_r2_scores):.3f} ± {np.nanstd(adj_r2_scores):.3f}')

# 회귀 문제이지만, 임계값을 정해 이진 분류로 변환하여 ROC 커브를 그릴 수 있습니다.
# 예시로, y값의 중앙값을 기준으로 1(전문의수 많음)/0(적음)으로 변환합니다.

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# y값 이진화 (중앙값 기준)
y_train_bin = (y_train >= y_train.median()).astype(int)
y_test_bin = (y_test >= y_train.median()).astype(int)  # train의 중앙값 기준

plt.figure(figsize=(10, 8))
for name, model in models.items():
    # 예측값을 그대로 사용 (회귀 결과)
    pred = model.predict(X_test)
    # ROC 커브 계산
    fpr, tpr, thresholds = roc_curve(y_test_bin, pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('모델별 ROC 커브 (전문의수 이진 분류)')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# 정리
# XGBoost: RMSE=1.59, MAE=1.25, R2=-0.181
# LightGBM: RMSE=1.32, MAE=1.02, R2=0.183
# CatBoost: RMSE=1.48, MAE=1.14, R2=-0.023
# RandomForest: RMSE=1.56, MAE=1.15, R2=-0.133
# LinearRegression: RMSE=1.32, MAE=1.06, R2=0.192

# 새로운 코드 추가
y_bin = (y >= y.median()).astype(int)

for name, model in models.items():
    # cross_val_predict로 전체 데이터에 대해 예측값 생성
    pred = cross_val_predict(model, X, y, cv=cv)
    auc_score = roc_auc_score(y_bin, pred)
    print(f'{name} CV ROC AUC: {auc_score:.3f}')

# 예시: XGB예측 외에 다른 예측값, 변화량, 비율 등 추가
X['XGB예측_log'] = np.log1p(X['XGB예측'])
# 병원명, 진료과를 대분류로 그룹화해서 새로운 컬럼 추가 등

# 1) 피처 선택/축소
selector = SelectKBest(score_func=f_regression, k=5)  # 상위 5개 피처만 선택

# 2) 차원 축소
pca = PCA(n_components=3)  # 3차원으로 축소

# 3) 정규화 (Ridge, Lasso)
ridge = Ridge(alpha=1.0)  # 정규화로 과적합 방지

def improved_modeling(X, y):
    """개선된 모델링 파이프라인"""
    
    # 1. 데이터 전처리
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. 피처 선택 (샘플 수 고려)
    n_features = min(5, X.shape[1]//2)  # 샘플 수의 절반 이하
    selector = SelectKBest(score_func=f_regression, k=n_features)
    X_selected = selector.fit_transform(X_scaled, y)
    
    # 3. 시계열 교차검증
    tscv = TimeSeriesSplit(n_splits=3)
    
    # 4. 모델 학습
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,  # 과적합 방지
        min_samples_split=5,
        random_state=42
    )
    
    # 5. 교차검증
    cv_scores = []
    for train_idx, val_idx in tscv.split(X_selected):
        X_train, X_val = X_selected[train_idx], X_selected[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)
        cv_scores.append({'RMSE': rmse, 'R2': r2})
    
    return cv_scores, rf, scaler, selector

# 사용 예시
# cv_results, model, scaler, selector = improved_modeling(X, y)

# 가중 평균 앙상블
def weighted_ensemble(models, weights, X):
    predictions = []
    for model in models:
        pred = model.predict(X)
        predictions.append(pred)
    
    weighted_pred = np.average(predictions, weights=weights, axis=0)
    return weighted_pred

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10)
    }
    
    model = RandomForestRegressor(**params, random_state=42)
    
    # 교차검증 수행
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = np.sqrt(-cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error'))
    
    return rmse_scores.mean()  # 평균 RMSE 반환

study = create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print(f"최적 하이퍼파라미터: {study.best_params}")
print(f"최적 RMSE: {study.best_value:.4f}")