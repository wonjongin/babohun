# -*- coding: utf-8 -*-
"""
Created on Sat Jun 28 20:43:03 2025
author: jenny

상병코드/지역 기반
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
data_csv = r"new_merged_data/다빈도 질환 환자 연령별 분포_순위추가_합계계산_값통일.csv"
mapping_csv = r"df_result2_with_심평원.csv"

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
# 2) 비모수 검정: Kruskal–Wallis + Dunn’s
# ----------------------------------------------------------------------
groups = [g['진료비'].values for _, g in df.groupby('상병코드') if len(g) >= 3]
H, p = stats.kruskal(*groups)
print(f"=== Kruskal–Wallis 검정: H={H:.4f}, p-value={p:.4e} ===")

dunn = sp.posthoc_dunn(df, val_col='진료비', group_col='상병코드', p_adjust='bonferroni')
print("=== Dunn’s post-hoc (Bonferroni) ===")
print(dunn)

# ----------------------------------------------------------------------
# 3) 분류 모델: 고비용 여부 예측
# ----------------------------------------------------------------------
thr = df['진료비'].quantile(0.75)
df['high_cost'] = (df['진료비'] >= thr).astype(int)

# Decision Tree
X_dt = pd.get_dummies(df[['상병코드']], prefix='', prefix_sep='')
y = df['high_cost']
X_tr_dt, X_te_dt, y_tr, y_te = train_test_split(
    X_dt, y, test_size=0.3, random_state=42, stratify=y
)
dt = DecisionTreeClassifier(max_depth=4, class_weight='balanced', random_state=42)
dt.fit(X_tr_dt, y_tr)
print("\n=== DecisionTreeClassifier ===")
print(classification_report(y_te, dt.predict(X_te_dt)))

# RandomForest & GradientBoosting
X_rf = pd.get_dummies(df[['상병코드', '지역']], dtype=int)
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

# 모델 저장
joblib.dump(dt, "dt_highcost_model.pkl")
joblib.dump(rf, "rf_highcost_model.pkl")
joblib.dump(gb, "gb_highcost_model.pkl")

# ----------------------------------------------------------------------
# 4) 회귀 모델: 진료비 직접 예측
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
print("\n=== 회귀 모델 평가 ===")
for name, m in [("DT", dtr), ("RF", rfr), ("GB", gbr)]:
    pred = m.predict(X_te_rg)
    print(f"{name} → MAE: {mean_absolute_error(y_te_rg, pred):.0f}천원, RMSE: {np.sqrt(mean_squared_error(y_te_rg, pred)):.0f}천원")
joblib.dump(dtr, "dtr_cost_regressor.pkl")
joblib.dump(rfr, "rfr_cost_regressor.pkl")
joblib.dump(gbr, "gbr_cost_regressor.pkl")

# ----------------------------------------------------------------------
# 5) 로그 스케일 기반 진료비 구간 예측
# ----------------------------------------------------------------------
# 5.1) 로그 스케일 구간 정의
min_v = df['진료비'].min()
max_v = df['진료비'].max()
bins = np.logspace(np.log10(min_v), np.log10(max_v), num=6)
# 5.2) 구간 클래스 할당
labels = pd.cut(df['진료비'], bins=bins, labels=False, include_lowest=True)
# 5.3) NaN & 희귀 구간 제거
valid_idx = labels.dropna().index
counts = labels.loc[valid_idx].value_counts().sort_index()
rare = counts[counts < 2].index
use_idx = valid_idx.difference(labels[labels.isin(rare)].index)
X_clean = X_reg.loc[use_idx]
y_clean = labels.loc[use_idx]
# 5.4) 학습/테스트 분할
X_tr, X_te, y_tr, y_te = train_test_split(X_clean, y_clean, test_size=0.3, random_state=42, stratify=y_clean)
# 5.5) 모델 학습 및 성능
lgb_clf = lgb.LGBMClassifier(objective='multiclass', num_class=len(y_clean.unique()), learning_rate=0.05, n_estimators=200, num_leaves=31, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5, verbosity=-1, seed=42)
lgb_clf.fit(X_tr, y_tr)
y_pred = lgb_clf.predict(X_te)
print("\n=== 로그 스케일 구간 분류 성능 ===")
print(classification_report(y_te, y_pred))
# 5.6) 대표 진료비 예측 함수
# 5.6) 대표 진료비 예측 함수
def predict_cost_bin(code, region, model, feat_cols, bins):
    """
    상병코드와 지역으로 로그 스케일 구간 클래스와
    대표 진료비를 예측하는 함수
    """
    # 입력 데이터프레임 생성 및 원-핫 인코딩
    df_in = pd.DataFrame([{'상병코드': code, '지역': region}])
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
bin_label, est_cost = predict_cost_bin(
    example_code, example_region,
    lgb_clf, feat_cols, bins
)
print(f"예측 구간: {bin_label}, 대표 진료비: {est_cost:.0f}천원")



import pandas as pd

# 1) 예측용 피처 리스트
feat_cols = X_reg.columns.tolist()

# 2) 전체 데이터에 대해 예측
#    predict_cost_bin 함수는 이미 정의되어 있다고 가정
preds = [
    predict_cost_bin(row['상병코드'], row['지역'], lgb_clf, feat_cols, bins)
    for _, row in df.iterrows()
]

# 3) 예측 결과를 df에 컬럼으로 추가
df['pred_bin'], df['pred_cost'] = zip(*preds)

# 4) CSV로 저장
output_path = r"src/formodel/진료비_구간예측결과.csv"
df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"예측 결과를 '{output_path}'에 저장했습니다.")
