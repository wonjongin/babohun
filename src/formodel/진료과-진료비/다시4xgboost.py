# -*- coding: utf-8 -*-
"""
Created on Sat Jun 28 20:43:03 2025
@author: jenny

상병코드/지역 기반
 1) 비모수 검정 (K-W + Dunn’s)
 2) 고비용 여부 분류 모델 (DT, RF, GB)
 3) 진료비 회귀 모델 (DT, RF, GB)
 4) 이상치(상위1%) 제거 후 회귀 모델 재학습
 5) LightGBM 회귀 모델 (로그 타깃 + CV)
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import scikit_posthocs as sp
import lightgbm as lgb
import joblib

from sklearn.model_selection   import train_test_split, KFold
from sklearn.tree              import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble          import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.metrics           import (
    classification_report,
    mean_absolute_error, mean_squared_error
)

# ──────────────────────────────────────────────────────────────────────────
# 1) 데이터 로드 & 전처리
# ──────────────────────────────────────────────────────────────────────────
data_csv    = r"C:/Users/jenny/babohun/new_merged_data/다빈도 질환 환자 연령별 분포_순위추가_합계계산_값통일.csv"
mapping_csv = r"C:/Users/jenny/babohun/df_result2_with_심평원.csv"

# 원본 읽고 “외래” 연인원 대체, “입원(실인원)” 삭제
ekqlseh = pd.read_csv(data_csv, encoding="utf-8-sig")
ekqlseh.loc[ekqlseh['구분'].str.contains('외래'), '연인원'] = ekqlseh['실인원']
ekqlseh = ekqlseh[ekqlseh['구분'] != '입원(실인원)']

# 필요 컬럼만, 특정 지역 제외, 칼럼명 통일
df = ekqlseh.drop(columns=['순위','상병명','실인원'])
df = df[~df['지역'].isin(['서울','대전','대구'])].copy()
df.rename(columns={'진료비(천원)':'진료비'}, inplace=True)

# 상병코드 ↔ 진료과 매핑 병합
mapping = pd.read_csv(mapping_csv, encoding="utf-8-sig")
df = df.merge(mapping[['상병코드','진료과']], on='상병코드', how='left')
df.dropna(subset=['진료과'], inplace=True)

# ──────────────────────────────────────────────────────────────────────────
# 2) 비모수 검정: Kruskal–Wallis + Dunn’s
# ──────────────────────────────────────────────────────────────────────────
print("=== Kruskal–Wallis 검정 ===")
groups = [g['진료비'].values for _, g in df.groupby('상병코드') if len(g)>=3]
H, p = stats.kruskal(*groups)
print(f"H={H:.4f}, p-value={p:.4e}\n")

print("=== Dunn’s post-hoc (Bonferroni) ===")
dunn = sp.posthoc_dunn(df, val_col='진료비', group_col='상병코드', p_adjust='bonferroni')
print(dunn)

# ──────────────────────────────────────────────────────────────────────────
# 3) 분류 모델링: 고비용 여부 예측
# ──────────────────────────────────────────────────────────────────────────
# — 레이블 생성 (상위25% → 1)
thr = df['진료비'].quantile(0.75)
df['high_cost'] = (df['진료비'] >= thr).astype(int)

# 3.1 DecisionTree (상병코드만)
X_dt = pd.get_dummies(df[['상병코드']], prefix='', prefix_sep='')
y    = df['high_cost']
X_tr_dt, X_te_dt, y_tr, y_te = train_test_split(
    X_dt, y, test_size=0.3, random_state=42, stratify=y
)
dt = DecisionTreeClassifier(max_depth=4, class_weight='balanced', random_state=42)
dt.fit(X_tr_dt, y_tr)
print("\n=== DecisionTreeClassifier ===")
print(classification_report(y_te, dt.predict(X_te_dt)))

# 3.2 RandomForest & GradientBoosting (상병코드+지역)
X_rf = pd.get_dummies(df[['상병코드','지역']], dtype=int)
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

print("\n=== RandomForestClassifier ===")
print(classification_report(y_te, rf.predict(X_te_rf)))
print("\n=== GradientBoostingClassifier ===")
print(classification_report(y_te, gb.predict(X_te_rf)))

print("\nRF 중요도 Top10:\n", pd.Series(rf.feature_importances_, index=X_rf.columns).nlargest(10))
print("\nGB 중요도 Top10:\n", pd.Series(gb.feature_importances_, index=X_rf.columns).nlargest(10))

# 모델 저장
joblib.dump(dt, "dt_highcost_model.pkl")
joblib.dump(rf, "rf_highcost_model.pkl")
joblib.dump(gb, "gb_highcost_model.pkl")

# ──────────────────────────────────────────────────────────────────────────
# 4) 회귀 모델링: 진료비 직접 예측
# ──────────────────────────────────────────────────────────────────────────
X_reg   = X_rf.copy()
y_reg   = df['진료비'].values
X_tr_rg, X_te_rg, y_tr_rg, y_te_rg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

dtr = DecisionTreeRegressor(max_depth=6, random_state=42)
rfr = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
gbr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)

for m in (dtr, rfr, gbr):
    m.fit(X_tr_rg, y_tr_rg)

print("\n=== 회귀 모델 평가 ===")
for name, m in [("DT",dtr),("RF",rfr),("GB",gbr)]:
    pred = m.predict(X_te_rg)
    mae  = mean_absolute_error(y_te_rg, pred)
    rmse = np.sqrt(mean_squared_error(y_te_rg, pred))
    print(f"{name} → MAE: {mae:.0f}천원, RMSE: {rmse:.0f}천원")

# 모델 저장
joblib.dump(dtr, "dtr_cost_regressor.pkl")
joblib.dump(rfr, "rfr_cost_regressor.pkl")
joblib.dump(gbr, "gbr_cost_regressor.pkl")

# ──────────────────────────────────────────────────────────────────────────
# 4.1) 이상치 제거 후 회귀 재학습 (상위1% 제외)
# ──────────────────────────────────────────────────────────────────────────
# 상위 1% 컷오프
outlier_thresh = np.quantile(y_tr_rg, 0.99)
mask_no       = y_tr_rg <= outlier_thresh
X_tr_no, y_tr_no = X_tr_rg[mask_no], y_tr_rg[mask_no]

# 재정의 후 학습
rfr_no = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
gbr_no = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
rfr_no.fit(X_tr_no, y_tr_no)
gbr_no.fit(X_tr_no, y_tr_no)

print("\n=== 이상치 제거 회귀 평가 ===")
for name, m in [("RF_no_outlier",rfr_no),("GB_no_outlier",gbr_no)]:
    p = m.predict(X_te_rg)
    print(f"{name} → MAE: {mean_absolute_error(y_te_rg, p):.0f}천원, RMSE: {np.sqrt(mean_squared_error(y_te_rg,p)):.0f}천원")

# ──────────────────────────────────────────────────────────────────────────
# 4.2) XGBoost 회귀 모델 (로그 타깃 + 이상치 제거 학습)
# ──────────────────────────────────────────────────────────────────────────
from xgboost import XGBRegressor

# (1) 로그 변환된 타깃 준비
y_tr_no_log = np.log1p(y_tr_no)
y_te_log    = np.log1p(y_te_rg)

# (2) XGBRegressor 정의
xgb = XGBRegressor(
    objective='reg:squarederror',
    learning_rate=0.05,
    n_estimators=1000,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

# (3) fit 호출 — callbacks 대신 early_stopping_rounds 사용
xgb.fit(
    X_tr_no, y_tr_no_log,
    eval_set=[(X_te_rg, y_te_log)],
    eval_metric='mae',
    early_stopping_rounds=50,
    verbose=False
)

# (4) 테스트셋 성능 확인 (원래 스케일로 복원)
pred_log_xgb = xgb.predict(X_te_rg)
pred_xgb     = np.expm1(pred_log_xgb)
print("\n=== XGBoost 회귀 성능 (이상치 제거) ===")
print(f"MAE: {mean_absolute_error(y_te_rg, pred_xgb):,.0f} 천원")
print(f"RMSE: {np.sqrt(mean_squared_error(y_te_rg, pred_xgb)):,.0f} 천원")

# (5) 모델 저장
joblib.dump(xgb, "xgb_cost_regressor.pkl")

# (6) 예시 예측
X_in = pd.DataFrame([{'상병코드': example_code, '지역': example_reg}])
X_in = pd.get_dummies(X_in[['상병코드','지역']], dtype=int)
X_in = X_in.reindex(columns=feat_cols, fill_value=0)
pred_xgb_example = np.expm1(xgb.predict(X_in)[0])
print(f"\n=== 예시 예측 (XGBoost) ===\nXGBoost 예상 진료비: {pred_xgb_example:,.0f} 천원")
