# -*- coding: utf-8 -*-
"""
Created on Sat Jun 28 20:43:03 2025
@author: jenny

상병코드/지역 기반
 1) 비모수 검정
 2) 고비용 여부 분류 모델
 3) 진료비 회귀 모델
 4) LightGBM 회귀 모델 (로그 타깃 + CV)
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import scikit_posthocs as sp
import lightgbm as lgb
import joblib

from sklearn.model_selection import train_test_split, KFold
from sklearn.tree       import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble   import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.metrics    import (
    classification_report,
    mean_absolute_error, mean_squared_error
)

# ----------------------------------------------------------------------
# 1) 데이터 로드 & 전처리
# ----------------------------------------------------------------------
data_csv    = r"C:/Users/jenny/babohun/new_merged_data/다빈도 질환 환자 연령별 분포_순위추가_합계계산_값통일.csv"
mapping_csv = r"C:/Users/jenny/babohun/df_result2_with_심평원.csv"

# 원본 불러오기
ekqlseh = pd.read_csv(data_csv, encoding="utf-8-sig")
# “외래”는 연인원=실인원, “입원(실인원)” 삭제
ekqlseh.loc[ekqlseh['구분'].str.contains('외래'), '연인원'] = ekqlseh['실인원']
ekqlseh = ekqlseh[ekqlseh['구분'] != '입원(실인원)']

# 필요 컬럼만, 서울·대전·대구 제외
df = ekqlseh.drop(columns=['순위','상병명','실인원'])
df = df[~df['지역'].isin(['서울','대전','대구'])].copy()
df.rename(columns={'진료비(천원)':'진료비'}, inplace=True)

# 상병코드 ↔ 진료과 매핑
mapping = pd.read_csv(mapping_csv, encoding="utf-8-sig")
df = df.merge(mapping[['상병코드','진료과']], on='상병코드', how='left')
df.dropna(subset=['진료과'], inplace=True)

# ----------------------------------------------------------------------
# 2) 비모수 검정: Kruskal–Wallis + Dunn’s
# ----------------------------------------------------------------------
print("=== Kruskal–Wallis 검정 ===")
groups = [g['진료비'].values for _, g in df.groupby('상병코드') if len(g)>=3]
H, p = stats.kruskal(*groups)
print(f"H={H:.4f}, p-value={p:.4e}\n")

print("=== Dunn’s post-hoc (Bonferroni) ===")
dunn = sp.posthoc_dunn(df, val_col='진료비', group_col='상병코드', p_adjust='bonferroni')
print(dunn)

# ----------------------------------------------------------------------
# 3) 분류 모델: 고비용 여부 예측
# ----------------------------------------------------------------------
# 레이블 생성 (상위 25% → 1, 나머지 0)
thr = df['진료비'].quantile(0.75)
df['high_cost'] = (df['진료비']>=thr).astype(int)

# -- 3.1 Decision Tree (상병코드만)
X_dt = pd.get_dummies(df[['상병코드']], prefix='', prefix_sep='')
y    = df['high_cost']
X_tr_dt, X_te_dt, y_tr, y_te = train_test_split(
    X_dt, y, test_size=0.3, random_state=42, stratify=y
)
dt = DecisionTreeClassifier(max_depth=4, class_weight='balanced', random_state=42)
dt.fit(X_tr_dt, y_tr)
print("\n=== DecisionTreeClassifier ===")
print(classification_report(y_te, dt.predict(X_te_dt)))

# -- 3.2 RandomForest & GB (상병코드+지역)
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
for name, m in [("DT",dtr),("RF",rfr),("GB",gbr)]:
    pred = m.predict(X_te_rg)
    mae  = mean_absolute_error(y_te_rg, pred)
    rmse = np.sqrt(mean_squared_error(y_te_rg, pred))
    print(f"{name} → MAE: {mae:.0f}천원, RMSE: {rmse:.0f}천원")

joblib.dump(dtr, "dtr_cost_regressor.pkl")
joblib.dump(rfr, "rfr_cost_regressor.pkl")
joblib.dump(gbr, "gbr_cost_regressor.pkl")

# 예측 함수 예시
def predict_cost(code, region, model, feat_cols):
    df_in = pd.DataFrame([{'상병코드':code,'지역':region}])
    X_in  = pd.get_dummies(df_in[['상병코드','지역']], dtype=int)
    X_in  = X_in.reindex(columns=feat_cols, fill_value=0)
    return model.predict(X_in)[0]

feat_cols    = X_reg.columns.tolist()
example_code = "M48"
example_reg  = "부산"
print("\n=== 예시 예측 (DT, RF, GB) ===")
for name, m in [("DT",dtr),("RF",rfr),("GB",gbr)]:
    c = predict_cost(example_code, example_reg, m, feat_cols)
    print(f"{name} 예상 진료비: {c:,.0f} 천원")
import pandas as pd
from sklearn.model_selection import train_test_split
############################################333333333
# 1) 진료비 10분위 구간화 (df에 cost_bin 컬럼 한 번만 생성)
df['cost_bin'] = pd.qcut(
    df['진료비'],
    q=8,
    labels=False,
    duplicates='drop'   # 경계 중복시 자동 구간 축소
)

# 2) 각 구간별 샘플 수 확인
counts = df['cost_bin'].value_counts().sort_index()
print(counts)

# 3) 너무 적은(예: 1개 이하) 샘플을 가진 구간 제거
rare_bins = counts[counts < 2].index
mask = ~df['cost_bin'].isin(rare_bins)

# 4) X_reg와 df의 인덱스가 일치한다고 가정하고 필터링
X_clean = X_reg.loc[mask]
y_clean = df.loc[mask, 'cost_bin']

# 5) stratify 적용하여 train/test 분할
X_tr, X_te, y_tr, y_te = train_test_split(
    X_clean,
    y_clean,
    test_size=0.3,
    random_state=42,
    stratify=y_clean
)

# 6) LightGBM 다중분류 모델 학습 예시
import lightgbm as lgb
from sklearn.metrics import classification_report

lgb_clf = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=len(y_clean.unique()),  # 실제 구간 수에 맞게
    learning_rate=0.05,
    n_estimators=200,
    num_leaves=31,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    seed=42
)
lgb_clf.fit(X_tr, y_tr)
y_pred = lgb_clf.predict(X_te)

print(classification_report(y_te, y_pred))

import pandas as pd

# 1) qcut 하면서 경계값까지 함께 얻기
#    df['진료비'] 는 천원 단위 정수라고 가정
labels, bins = pd.qcut(
    df['진료비'],
    q=8,
    labels=False,
    retbins=True,
    duplicates='drop'
)

# 2) bins 는 구간 경계값 배열 (len = 실제 구간 수 + 1)
print("경계값(bins):", bins)

# 3) IntervalIndex 로 변환해 보기
intervals = pd.IntervalIndex.from_breaks(bins)

# 4) 클래스별 범위 출력
for k, interval in enumerate(intervals):
    left  = int(interval.left)   # 구간 하한
    right = int(interval.right)  # 구간 상한
    # 마지막 구간은 오른쪽 포함
    right_char = '≤' if k == len(intervals)-1 else '<'
    print(f"클래스 {k}: {left}천원 ≤ 진료비 {right_char} {right}천원")


# 1) qcut 하면서 경계값까지 함께 얻기
#    df['진료비'] 는 천원 단위 정수라고 가정
labels, bins = pd.qcut(
    df['진료비'],
    q=4,
    labels=False,
    retbins=True,
    duplicates='drop'
)

# 2) bins 는 구간 경계값 배열 (len = 실제 구간 수 + 1)
print("경계값(bins):", bins)

# 3) IntervalIndex 로 변환해 보기
intervals = pd.IntervalIndex.from_breaks(bins)

# 4) 클래스별 범위 출력
for k, interval in enumerate(intervals):
    left  = int(interval.left)   # 구간 하한
    right = int(interval.right)  # 구간 상한
    # 마지막 구간은 오른쪽 포함
    right_char = '≤' if k == len(intervals)-1 else '<'
    print(f"클래스 {k}: {left}천원 ≤ 진료비 {right_char} {right}천원")
