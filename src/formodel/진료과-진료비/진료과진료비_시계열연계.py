# -*- coding: utf-8 -*-
"""
Created on Sat Jun 28 20:43:03 2025
author: jenny

상병코드/지역 기반 + 시계열 예측 데이터 연계
 1) 비모수 검정
 2) 고비용 여부 분류 모델
 3) 진료비 회귀 모델
 4) LightGBM 회귀 모델 (로그 타깃 + CV)
 5) 로그 스케일 기반 진료비 구간 예측
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import scikit_posthocs as sp
import lightgbm as lgb
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.metrics import (
    classification_report,
    mean_absolute_error, mean_squared_error
)

# ----------------------------------------------------------------------
# 1) 데이터 로드 & 전처리
# ----------------------------------------------------------------------
print("=== 데이터 로딩 시작 ===")

data_csv = r"new_merged_data/다빈도 질환 환자 연령별 분포_순위추가_합계계산_값통일.csv"
mapping_csv = r"df_result2_with_심평원.csv"

# 시계열 예측 데이터 로드 (성능 향상을 위해)
try:
    df_pred = pd.read_csv('analysis_data/병원별_진료과별_입원외래_통합_시계열예측결과_개선.csv')
    print("✅ 시계열 예측 데이터 로드 완료")
except:
    print("⚠️ 시계열 예측 데이터 없음")
    df_pred = None

ekqlseh = pd.read_csv(data_csv, encoding="utf-8-sig")
ekqlseh.loc[ekqlseh['구분'].str.contains('외래'), '연인원'] = ekqlseh['실인원']
ekqlseh = ekqlseh[ekqlseh['구분'] != '입원(실인원)']

df = ekqlseh.drop(columns=['순위', '상병명', '실인원'])
df = df[~df['지역'].isin(['서울', '대전', '대구'])].copy()
df.rename(columns={'진료비(천원)': '진료비'}, inplace=True)

mapping = pd.read_csv(mapping_csv, encoding="utf-8-sig")
df = df.merge(mapping[['상병코드', '진료과']], on='상병코드', how='left')
df.dropna(subset=['진료과'], inplace=True)

# ----------------------------------------------------------------------
# 2) 시계열 예측 데이터 기반 추가 피처 생성
# ----------------------------------------------------------------------
print("=== 시계열 예측 데이터 기반 피처 생성 ===")

if df_pred is not None:
    # 시계열 예측 데이터에서 상병코드별 예측값 통합
    pred_summary = df_pred.groupby(['진료과', '연도']).agg({
        'ARIMA예측': 'mean',
        'RF예측': 'mean',
        'XGB예측': 'mean',
        '실제값': 'mean'
    }).reset_index()
    
    pred_summary.columns = ['진료과', '연도', 'ARIMA예측_평균', 'RF예측_평균', 'XGB예측_평균', '실제값_평균']
    
    # 진료과별로 매핑 (연도 정보가 없으면 전체 평균 사용)
    df = df.merge(pred_summary, on='진료과', how='left')
    
    # 예측값 관련 피처 생성
    df['예측값_평균'] = df[['ARIMA예측_평균', 'RF예측_평균', 'XGB예측_평균']].mean(axis=1)
    df['예측값_표준편차'] = df[['ARIMA예측_평균', 'RF예측_평균', 'XGB예측_평균']].std(axis=1)
    df['가중예측값'] = (0.2 * df['ARIMA예측_평균'] + 0.3 * df['RF예측_평균'] + 0.5 * df['XGB예측_평균'])
    
    # 예측 정확도 지표
    df['ARIMA_오차'] = abs(df['ARIMA예측_평균'] - df['실제값_평균'])
    df['RF_오차'] = abs(df['RF예측_평균'] - df['실제값_평균'])
    df['XGB_오차'] = abs(df['XGB예측_평균'] - df['실제값_평균'])
    
    # 로그 변환
    df['ARIMA예측_log'] = np.log1p(np.abs(df['ARIMA예측_평균']))
    df['RF예측_log'] = np.log1p(np.abs(df['RF예측_평균']))
    df['XGB예측_log'] = np.log1p(np.abs(df['XGB예측_평균']))
    df['실제값_log'] = np.log1p(np.abs(df['실제값_평균']))
    
    # 비율 피처
    df['ARIMA_비율'] = np.where(df['실제값_평균'] != 0, df['ARIMA예측_평균'] / df['실제값_평균'], 1.0)
    df['RF_비율'] = np.where(df['실제값_평균'] != 0, df['RF예측_평균'] / df['실제값_평균'], 1.0)
    df['XGB_비율'] = np.where(df['실제값_평균'] != 0, df['XGB예측_평균'] / df['실제값_평균'], 1.0)
    
    # NaN 값 처리
    df = df.fillna(0)
    
    print(f"시계열 예측 데이터 기반 추가 피처 생성 완료")
    print(f"추가된 피처 수: {len(['예측값_평균', '예측값_표준편차', '가중예측값', 'ARIMA_오차', 'RF_오차', 'XGB_오차', 'ARIMA예측_log', 'RF예측_log', 'XGB예측_log', '실제값_log', 'ARIMA_비율', 'RF_비율', 'XGB_비율'])}개")
else:
    # 시계열 데이터가 없는 경우 기본값 설정
    df['ARIMA예측_평균'] = 0
    df['RF예측_평균'] = 0
    df['XGB예측_평균'] = 0
    df['실제값_평균'] = 0
    df['예측값_평균'] = 0
    df['예측값_표준편차'] = 0
    df['가중예측값'] = 0
    df['ARIMA_오차'] = 0
    df['RF_오차'] = 0
    df['XGB_오차'] = 0
    df['ARIMA예측_log'] = 0
    df['RF예측_log'] = 0
    df['XGB예측_log'] = 0
    df['실제값_log'] = 0
    df['ARIMA_비율'] = 1.0
    df['RF_비율'] = 1.0
    df['XGB_비율'] = 1.0

# ----------------------------------------------------------------------
# 3) 비모수 검정: Kruskal–Wallis + Dunn's
# ----------------------------------------------------------------------
groups = [g['진료비'].values for _, g in df.groupby('상병코드') if len(g) >= 3]
H, p = stats.kruskal(*groups)
print(f"=== Kruskal–Wallis 검정: H={H:.4f}, p-value={p:.4e} ===")

dunn = sp.posthoc_dunn(df, val_col='진료비', group_col='상병코드', p_adjust='bonferroni')
print("=== Dunn's post-hoc (Bonferroni) ===")
print(dunn)

# ----------------------------------------------------------------------
# 4) 분류 모델: 고비용 여부 예측 (시계열 데이터 포함)
# ----------------------------------------------------------------------
thr = df['진료비'].quantile(0.75)
df['high_cost'] = (df['진료비'] >= thr).astype(int)

# Decision Tree (시계열 데이터 포함)
X_dt = pd.get_dummies(df[['상병코드', '예측값_평균', '가중예측값']], prefix='', prefix_sep='')
y = df['high_cost']
X_tr_dt, X_te_dt, y_tr, y_te = train_test_split(
    X_dt, y, test_size=0.3, random_state=42, stratify=y
)
dt = DecisionTreeClassifier(max_depth=4, class_weight='balanced', random_state=42)
dt.fit(X_tr_dt, y_tr)
print("\n=== DecisionTreeClassifier (시계열 데이터 포함) ===")
print(classification_report(y_te, dt.predict(X_te_dt)))

# RandomForest & GradientBoosting (시계열 데이터 포함)
X_rf = pd.get_dummies(df[['상병코드', '지역', '예측값_평균', '가중예측값', 'ARIMA_오차', 'RF_오차', 'XGB_오차']], dtype=int)
X_tr_rf, X_te_rf, _, _ = train_test_split(
    X_rf, y, test_size=0.3, random_state=42, stratify=y
)
rf = RandomForestClassifier(
    n_estimators=200, max_depth=6,
    class_weight='balanced', random_state=42, n_jobs=-1
)
gb = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.05,
    max_depth=4, random_state=42
)
rf.fit(X_tr_rf, y_tr)
gb.fit(X_tr_rf, y_tr)
print("\n=== RandomForestClassifier (시계열 데이터 포함) ===")
print(classification_report(y_te, rf.predict(X_te_rf)))
print("\n=== GradientBoostingClassifier (시계열 데이터 포함) ===")
print(classification_report(y_te, gb.predict(X_te_rf)))

# ----------------------------------------------------------------------
# 5) 회귀 모델: 진료비 직접 예측 (시계열 데이터 포함)
# ----------------------------------------------------------------------
X_reg = X_rf.copy()
y_reg = df['진료비'].values
X_tr_rg, X_te_rg, y_tr_rg, y_te_rg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

dtr = DecisionTreeRegressor(max_depth=6, random_state=42)
rfr = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
gbr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
for m in (dtr, rfr, gbr):
    m.fit(X_tr_rg, y_tr_rg)
print("\n=== 회귀 모델 평가 (시계열 데이터 포함) ===")
for name, m in [("DT", dtr), ("RF", rfr), ("GB", gbr)]:
    pred = m.predict(X_te_rg)
    print(f"{name} → MAE: {mean_absolute_error(y_te_rg, pred):.0f}천원, RMSE: {np.sqrt(mean_squared_error(y_te_rg, pred)):.0f}천원")

# ----------------------------------------------------------------------
# 6) 로그 스케일 기반 진료비 구간 예측 (시계열 데이터 포함)
# ----------------------------------------------------------------------
# 6.1) 로그 스케일 구간 정의
min_v = df['진료비'].min()
max_v = df['진료비'].max()
bins = np.logspace(np.log10(min_v), np.log10(max_v), num=6)
# 6.2) 구간 클래스 할당
labels = pd.cut(df['진료비'], bins=bins, labels=False, include_lowest=True)
# 6.3) NaN & 희귀 구간 제거
valid_idx = labels.dropna().index
counts = labels.loc[valid_idx].value_counts().sort_index()
rare = counts[counts < 2].index
use_idx = valid_idx.difference(labels[labels.isin(rare)].index)
X_clean = X_reg.loc[use_idx]
y_clean = labels.loc[use_idx]
# 6.4) 학습/테스트 분할
X_tr, X_te, y_tr, y_te = train_test_split(X_clean, y_clean, test_size=0.3, random_state=42, stratify=y_clean)
# 6.5) 모델 학습 및 성능
lgb_clf = lgb.LGBMClassifier(objective='multiclass', num_class=len(y_clean.unique()), learning_rate=0.05, n_estimators=200, num_leaves=31, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5, verbosity=-1, seed=42)
lgb_clf.fit(X_tr, y_tr)
y_pred = lgb_clf.predict(X_te)
print("\n=== 로그 스케일 구간 분류 성능 (시계열 데이터 포함) ===")
print(classification_report(y_te, y_pred))

# 6.6) 대표 진료비 예측 함수
def predict_cost_bin(code, region, model, feat_cols, bins, pred_features=None):
    """
    상병코드와 지역으로 로그 스케일 구간 클래스와
    대표 진료비를 예측하는 함수 (시계열 데이터 포함)
    """
    # 입력 데이터프레임 생성 및 원-핫 인코딩
    df_in = pd.DataFrame([{'상병코드': code, '지역': region}])
    
    # 시계열 예측 데이터가 있으면 추가
    if pred_features is not None:
        for key, value in pred_features.items():
            df_in[key] = value
    
    X_in = pd.get_dummies(df_in, columns=['상병코드', '지역'], dtype=int)
    X_in = X_in.reindex(columns=feat_cols, fill_value=0)

    # 클래스 예측
    bin_pred_raw = model.predict(X_in)[0]
    bin_idx = int(bin_pred_raw)

    # 클래스별 대표값 추출 (중앙값)
    midpoint = (bins[bin_idx] + bins[bin_idx + 1]) / 2
    return bin_idx, midpoint

# 예시 사용:
feat_cols = X_reg.columns.tolist()
example_code, example_region = 'M48', '부산'
# 시계열 예측값 예시 (실제로는 해당 상병코드의 예측값을 사용)
pred_features = {
    '예측값_평균': 1000,
    '가중예측값': 950,
    'ARIMA_오차': 50,
    'RF_오차': 30,
    'XGB_오차': 20
}
bin_label, est_cost = predict_cost_bin(
    example_code, example_region,
    lgb_clf, feat_cols, bins, pred_features
)
print(f"예측 구간: {bin_label}, 대표 진료비: {est_cost:.0f}천원")

# ----------------------------------------------------------------------
# 7) 결과 저장
# ----------------------------------------------------------------------
print("\n=== 결과 저장 시작 ===")

# 결과 저장 디렉토리 생성
results_dir = "model_results_진료과진료비_시계열"
os.makedirs(results_dir, exist_ok=True)

# 모델 저장
joblib.dump(dt, f"{results_dir}/dt_highcost_model_timeseries.pkl")
joblib.dump(rf, f"{results_dir}/rf_highcost_model_timeseries.pkl")
joblib.dump(gb, f"{results_dir}/gb_highcost_model_timeseries.pkl")
joblib.dump(dtr, f"{results_dir}/dtr_cost_regressor_timeseries.pkl")
joblib.dump(rfr, f"{results_dir}/rfr_cost_regressor_timeseries.pkl")
joblib.dump(gbr, f"{results_dir}/gbr_cost_regressor_timeseries.pkl")
joblib.dump(lgb_clf, f"{results_dir}/lgb_cost_bin_classifier_timeseries.pkl")

# 전체 데이터에 대해 예측
preds = []
for _, row in df.iterrows():
    # 해당 상병코드의 시계열 예측값 가져오기
    pred_features = {
        '예측값_평균': row.get('예측값_평균', 0),
        '가중예측값': row.get('가중예측값', 0),
        'ARIMA_오차': row.get('ARIMA_오차', 0),
        'RF_오차': row.get('RF_오차', 0),
        'XGB_오차': row.get('XGB_오차', 0)
    }
    
    try:
        bin_label, est_cost = predict_cost_bin(
            row['상병코드'], row['지역'],
            lgb_clf, feat_cols, bins, pred_features
        )
        preds.append((bin_label, est_cost))
    except:
        preds.append((0, 0))

# 예측 결과를 df에 컬럼으로 추가
df['pred_bin_timeseries'], df['pred_cost_timeseries'] = zip(*preds)

# CSV로 저장
output_path = f"{results_dir}/진료비_구간예측결과_시계열연계.csv"
df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"예측 결과를 '{output_path}'에 저장했습니다.")
print(f"모든 결과가 '{results_dir}' 디렉토리에 저장되었습니다!") 