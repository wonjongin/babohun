# -*- coding: utf-8 -*-
"""
Created on Sat Jun 28 20:43:03 2025
@author: jenny

상병코드/지역 기반 고비용 분류 모델 & 진료비 회귀 모델
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import scikit_posthocs as sp
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error
import joblib

# -----------------------------------------------------------------------------
# 1) 데이터 로드 & 전처리
# -----------------------------------------------------------------------------
ekqlseh = pd.read_csv("C:/Users/jenny/babohun/new_merged_data/다빈도 질환 환자 연령별 분포_순위추가_합계계산_값통일.csv",encoding="utf-8-sig")
# 외래인원으로 연인원 대체, 입원(실인원) 삭제
ekqlseh.loc[ekqlseh['구분'].str.contains('외래'), '연인원'] = ekqlseh['실인원']
ekqlseh = ekqlseh[ekqlseh['구분'] != '입원(실인원)']

# 필요 컬럼 추리고 서울·대전·대구 제외
df = ekqlseh.drop(columns=['순위', '상병명', '실인원'])
df = df[~df['지역'].isin(['서울', '대전', '대구'])].copy()
df.rename(columns={'진료비(천원)': '진료비'}, inplace=True)

# 상병코드 ↔ 진료과 매핑 병합
mapping = pd.read_csv("C:/Users/jenny/babohun/df_result2_with_심평원.csv",encoding="utf-8-sig")
df = (df.merge(mapping[['상병코드', '진료과']], on='상병코드', how='left').dropna(subset=['진료과']))

# -----------------------------------------------------------------------------
# 2) 비모수 검정 (Kruskal–Wallis + Dunn’s)
# -----------------------------------------------------------------------------
print("=== 정규성 검정 (Shapiro–Wilk by 상병코드) ===")
for code, grp in df.groupby('상병코드'):
    vals = grp['진료비'].dropna().values
    if len(vals) < 3:
        print(f"{code}: n={len(vals)} (<3) → 건너뜀")
        continue
    if len(vals) > 500:
        vals = np.random.choice(vals, 500, replace=False)
    W, p = stats.shapiro(vals)
    print(f"{code}: W={W:.4f}, p={p:.4e} (n={len(grp)})")

print("\n=== 등분산성 검정 (Levene) ===")
groups = [g['진료비'].dropna().values for _, g in df.groupby('상병코드') if len(g)>=2]
H, p = stats.levene(*groups, center='median')
print(f"Levene H={H:.4f}, p={p:.4e}")

print("\n=== Kruskal–Wallis ===")
groups = [g['진료비'].dropna().values for _, g in df.groupby('상병코드') if len(g)>=3]
H, p = stats.kruskal(*groups)
print(f"H={H:.4f}, p={p:.4e}")

print("\n=== Dunn’s post-hoc (Bonferroni) ===")
posthoc = sp.posthoc_dunn(df, val_col='진료비', group_col='상병코드', p_adjust='bonferroni')
print(posthoc)

# -----------------------------------------------------------------------------
# 3) 분류 모델링: ‘고비용 여부’ 예측
# -----------------------------------------------------------------------------
# 레이블 생성
threshold = df['진료비'].quantile(0.75)
df['high_cost'] = (df['진료비'] >= threshold).astype(int)

# Decision Tree (상병코드만)
X_dt = pd.get_dummies(df[['상병코드']], prefix='', prefix_sep='')
y = df['high_cost']
X_train_dt, X_test_dt, y_train, y_test = train_test_split(
    X_dt, y, test_size=0.3, random_state=42, stratify=y
)
dt = DecisionTreeClassifier(max_depth=4, class_weight='balanced', random_state=42)
dt.fit(X_train_dt, y_train)
print("\n=== DecisionTreeClassifier ===")
print(classification_report(y_test, dt.predict(X_test_dt)))

# RandomForest & GB (상병코드+지역)
X_rf = pd.get_dummies(df[['상병코드', '지역']], dtype=int)
X_train_rf, X_test_rf, _, _ = train_test_split(
    X_rf, y, test_size=0.3, random_state=42, stratify=y
)
rf = RandomForestClassifier(n_estimators=200, max_depth=6,
                            class_weight='balanced', random_state=42, n_jobs=-1)
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                max_depth=4, random_state=42)
rf.fit(X_train_rf, y_train)
gb.fit(X_train_rf, y_train)

print("\n=== RandomForestClassifier ===")
print(classification_report(y_test, rf.predict(X_test_rf)))
print("\n=== GradientBoostingClassifier ===")
print(classification_report(y_test, gb.predict(X_test_rf)))

# 중요도 출력
imp_rf = pd.Series(rf.feature_importances_, index=X_rf.columns).nlargest(10)
imp_gb = pd.Series(gb.feature_importances_, index=X_rf.columns).nlargest(10)
print("\nRF 중요도 Top10:\n", imp_rf)
print("\nGB 중요도 Top10:\n", imp_gb)

# 모델 저장
joblib.dump(dt, "dt_highcost_model.pkl")
joblib.dump(rf, "rf_highcost_model.pkl")
joblib.dump(gb, "gb_highcost_model.pkl")

# -----------------------------------------------------------------------------
# 4) 회귀 모델링: 진료비 직접 예측
# -----------------------------------------------------------------------------
X_reg = X_rf.copy()
y_reg = df['진료비'].values
X_train_rg, X_test_rg, y_train_rg, y_test_rg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

dtr = DecisionTreeRegressor(max_depth=6, random_state=42)
rfr = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
gbr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)

for model in (dtr, rfr, gbr):
    model.fit(X_train_rg, y_train_rg)

print("\n=== 회귀 모델 평가 ===")
for name, model in [("DT", dtr), ("RF", rfr), ("GB", gbr)]:
    pred = model.predict(X_test_rg)
    mae = mean_absolute_error(y_test_rg, pred)
    rmse = np.sqrt(mean_squared_error(y_test_rg, pred))
    print(f"{name} → MAE: {mae:.0f}천원, RMSE: {rmse:.0f}천원")

# 모델 저장
joblib.dump(dtr, "dtr_cost_regressor.pkl")
joblib.dump(rfr, "rfr_cost_regressor.pkl")
joblib.dump(gbr, "gbr_cost_regressor.pkl")

# -----------------------------------------------------------------------------
# 5) 예측 함수 예시
# -----------------------------------------------------------------------------
def predict_cost(code, region, model, feature_cols):
    df_in = pd.DataFrame([{'상병코드': code, '지역': region}])
    X_in = pd.get_dummies(df_in[['상병코드','지역']], dtype=int)
    X_in = X_in.reindex(columns=feature_cols, fill_value=0)
    return model.predict(X_in)[0]

feature_cols = X_reg.columns.tolist()
print("\n=== 예시 예측 ===")
for name, model in [("DT", dtr), ("RF", rfr), ("GB", gbr)]:
    cost = predict_cost("M48", "부산", model, feature_cols)
    print(f"{name} 예상 진료비: {cost:,.0f} 천원")

# 이미 학습하신 GradientBoostingRegressor(gbr) 예시
pred_cost = gbr.predict(X_in)[0]
print(f"예상 진료비: {pred_cost:,.0f} 천원")
import numpy as np

def bootstrap_preds(model_cls, X_train, y_train, X_in, n_boot=100):
    preds = []
    n = len(X_train)
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        m = model_cls.fit(X_train.iloc[idx], y_train[idx])
        preds.append(m.predict(X_in)[0])
    arr = np.array(preds)
    return np.percentile(arr, [2.5, 50, 97.5])  # lower, median, upper

# 예시: RandomForestRegressor 로 95% 예측 구간
from sklearn.ensemble import RandomForestRegressor
pi_low, pi_med, pi_high = bootstrap_preds(
    RandomForestRegressor(n_estimators=50, random_state=0),
    X_train, y_train,
    X_in=pd.DataFrame([{'상병코드_M48':1, '지역_부산':1}]),
    n_boot=200
)
print(f"95% 예측 구간: {pi_low:,.0f} ~ {pi_high:,.0f} 천원")
from sklearn.ensemble import GradientBoostingRegressor

# 10%, 50%, 90% 분위수 모델
qrs = {
    q: GradientBoostingRegressor(loss='quantile', alpha=q, n_estimators=200, random_state=42)
    for q in [0.1, 0.5, 0.9]
}
for q, model in qrs.items():
    model.fit(X_train, y_train)

# 예측값
X_new = X_in  # 위에서 만든 상병코드·지역 더미
preds = {q: m.predict(X_new)[0] for q,m in qrs.items()}
print("10% :", preds[0.1], "50% :", preds[0.5], "90% :", preds[0.9])
