# -*- coding: utf-8 -*-
"""
Created on Sat Jun 28 20:43:03 2025
author: jenny

상병코드/지역 기반 + 연령지역진료과 데이터 연계
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

# 연령지역진료과 데이터 로드 (성능 향상을 위해)
try:
    df_age_region = pd.read_csv('model_results_연령지역_진료과/Stacking_prediction_results_detailed.csv')
    print("✅ 연령지역 진료과 데이터 로드 완료")
except:
    print("⚠️ 연령지역 진료과 데이터 없음")
    df_age_region = None

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
# 2) 연령지역진료과 데이터 기반 추가 피처 생성
# ----------------------------------------------------------------------
print("=== 연령지역진료과 데이터 기반 피처 생성 ===")

if df_age_region is not None:
    # 연령지역진료과 데이터에서 진료과별 통계 정보 추출
    department_stats = df_age_region.groupby('y_actual').agg({
        'top1_probability': 'mean',
        'confidence': 'mean',
        'sample_weight': 'sum',
        'age_num': 'mean',
        'is_major_city': 'mean'
    }).reset_index()
    
    department_stats.columns = ['진료과', '평균확률', '평균신뢰도', '총샘플수', '평균연령', '대도시비율']
    
    # 진료과별로 매핑
    df = df.merge(department_stats, on='진료과', how='left')
    
    # 지역별 통계 정보 추출
    region_stats = df_age_region.groupby('지역').agg({
        'top1_probability': 'mean',
        'confidence': 'mean',
        'sample_weight': 'sum',
        'age_num': 'mean'
    }).reset_index()
    
    region_stats.columns = ['지역', '지역평균확률', '지역평균신뢰도', '지역총샘플수', '지역평균연령']
    
    # 지역별로 매핑
    df = df.merge(region_stats, on='지역', how='left')
    
    # 연령대별 통계 정보 추출
    age_stats = df_age_region.groupby('age_group').agg({
        'top1_probability': 'mean',
        'confidence': 'mean',
        'sample_weight': 'sum'
    }).reset_index()
    
    age_stats.columns = ['연령대', '연령대평균확률', '연령대평균신뢰도', '연령대총샘플수']
    
    # 연령대 정보가 있다면 매핑 (없으면 기본값 사용)
    if 'age_group' in df.columns:
        df = df.merge(age_stats, on='연령대', how='left')
    else:
        # 연령대 정보가 없으면 전체 평균값 사용
        df['연령대평균확률'] = age_stats['연령대평균확률'].mean()
        df['연령대평균신뢰도'] = age_stats['연령대평균신뢰도'].mean()
        df['연령대총샘플수'] = age_stats['연령대총샘플수'].mean()
    
    # 상호작용 피처 생성
    df['진료과_지역_상호작용'] = df['평균확률'] * df['지역평균확률']
    df['진료과_연령대_상호작용'] = df['평균확률'] * df['연령대평균확률']
    df['지역_연령대_상호작용'] = df['지역평균확률'] * df['연령대평균확률']
    
    # 복합 신뢰도 지표
    df['종합신뢰도'] = (df['평균신뢰도'] + df['지역평균신뢰도'] + df['연령대평균신뢰도']) / 3
    
    # 로그 변환
    df['총샘플수_log'] = np.log1p(df['총샘플수'])
    df['지역총샘플수_log'] = np.log1p(df['지역총샘플수'])
    df['연령대총샘플수_log'] = np.log1p(df['연령대총샘플수'])
    
    # NaN 값 처리
    df = df.fillna(0)
    
    print(f"연령지역진료과 데이터 기반 추가 피처 생성 완료")
    print(f"추가된 피처 수: {len(['평균확률', '평균신뢰도', '총샘플수', '평균연령', '대도시비율', '지역평균확률', '지역평균신뢰도', '지역총샘플수', '지역평균연령', '연령대평균확률', '연령대평균신뢰도', '연령대총샘플수', '진료과_지역_상호작용', '진료과_연령대_상호작용', '지역_연령대_상호작용', '종합신뢰도', '총샘플수_log', '지역총샘플수_log', '연령대총샘플수_log'])}개")
else:
    # 연령지역진료과 데이터가 없는 경우 기본값 설정
    df['평균확률'] = 0
    df['평균신뢰도'] = 0
    df['총샘플수'] = 0
    df['평균연령'] = 0
    df['대도시비율'] = 0
    df['지역평균확률'] = 0
    df['지역평균신뢰도'] = 0
    df['지역총샘플수'] = 0
    df['지역평균연령'] = 0
    df['연령대평균확률'] = 0
    df['연령대평균신뢰도'] = 0
    df['연령대총샘플수'] = 0
    df['진료과_지역_상호작용'] = 0
    df['진료과_연령대_상호작용'] = 0
    df['지역_연령대_상호작용'] = 0
    df['종합신뢰도'] = 0
    df['총샘플수_log'] = 0
    df['지역총샘플수_log'] = 0
    df['연령대총샘플수_log'] = 0

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
# 4) 분류 모델: 고비용 여부 예측 (연령지역진료과 데이터 포함)
# ----------------------------------------------------------------------
thr = df['진료비'].quantile(0.75)
df['high_cost'] = (df['진료비'] >= thr).astype(int)

# Decision Tree (연령지역진료과 데이터 포함)
X_dt = pd.get_dummies(df[['상병코드', '평균확률', '종합신뢰도']], prefix='', prefix_sep='')
y = df['high_cost']
X_tr_dt, X_te_dt, y_tr, y_te = train_test_split(
    X_dt, y, test_size=0.3, random_state=42, stratify=y
)
dt = DecisionTreeClassifier(max_depth=4, class_weight='balanced', random_state=42)
dt.fit(X_tr_dt, y_tr)
print("\n=== DecisionTreeClassifier (연령지역진료과 데이터 포함) ===")
print(classification_report(y_te, dt.predict(X_te_dt)))

# RandomForest & GradientBoosting (연령지역진료과 데이터 포함)
X_rf = pd.get_dummies(df[['상병코드', '지역', '평균확률', '종합신뢰도', '진료과_지역_상호작용', '총샘플수_log']], dtype=int)
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
print("\n=== RandomForestClassifier (연령지역진료과 데이터 포함) ===")
print(classification_report(y_te, rf.predict(X_te_rf)))
print("\n=== GradientBoostingClassifier (연령지역진료과 데이터 포함) ===")
print(classification_report(y_te, gb.predict(X_te_rf)))

# ----------------------------------------------------------------------
# 5) 회귀 모델: 진료비 직접 예측 (연령지역진료과 데이터 포함)
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
print("\n=== 회귀 모델 평가 (연령지역진료과 데이터 포함) ===")
for name, m in [("DT", dtr), ("RF", rfr), ("GB", gbr)]:
    pred = m.predict(X_te_rg)
    print(f"{name} → MAE: {mean_absolute_error(y_te_rg, pred):.0f}천원, RMSE: {np.sqrt(mean_squared_error(y_te_rg, pred)):.0f}천원")

# ----------------------------------------------------------------------
# 6) 로그 스케일 기반 진료비 구간 예측 (연령지역진료과 데이터 포함)
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
print("\n=== 로그 스케일 구간 분류 성능 (연령지역진료과 데이터 포함) ===")
print(classification_report(y_te, y_pred))

# 6.6) 대표 진료비 예측 함수
def predict_cost_bin(code, region, model, feat_cols, bins, age_region_features=None):
    """
    상병코드와 지역으로 로그 스케일 구간 클래스와
    대표 진료비를 예측하는 함수 (연령지역진료과 데이터 포함)
    """
    # 입력 데이터프레임 생성 및 원-핫 인코딩
    df_in = pd.DataFrame([{'상병코드': code, '지역': region}])
    
    # 연령지역진료과 데이터가 있으면 추가
    if age_region_features is not None:
        for key, value in age_region_features.items():
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
# 연령지역진료과 예측값 예시 (실제로는 해당 상병코드의 통계값을 사용)
age_region_features = {
    '평균확률': 0.8,
    '종합신뢰도': 0.75,
    '진료과_지역_상호작용': 0.6,
    '총샘플수_log': 8.5
}
bin_label, est_cost = predict_cost_bin(
    example_code, example_region,
    lgb_clf, feat_cols, bins, age_region_features
)
print(f"예측 구간: {bin_label}, 대표 진료비: {est_cost:.0f}천원")

# ----------------------------------------------------------------------
# 7) 결과 저장
# ----------------------------------------------------------------------
print("\n=== 결과 저장 시작 ===")

# 결과 저장 디렉토리 생성
results_dir = "model_results_진료과진료비_연령지역진료과"
os.makedirs(results_dir, exist_ok=True)

# 모델 저장
joblib.dump(dt, f"{results_dir}/dt_highcost_model_age_region.pkl")
joblib.dump(rf, f"{results_dir}/rf_highcost_model_age_region.pkl")
joblib.dump(gb, f"{results_dir}/gb_highcost_model_age_region.pkl")
joblib.dump(dtr, f"{results_dir}/dtr_cost_regressor_age_region.pkl")
joblib.dump(rfr, f"{results_dir}/rfr_cost_regressor_age_region.pkl")
joblib.dump(gbr, f"{results_dir}/gbr_cost_regressor_age_region.pkl")
joblib.dump(lgb_clf, f"{results_dir}/lgb_cost_bin_classifier_age_region.pkl")

# 전체 데이터에 대해 예측
preds = []
for _, row in df.iterrows():
    # 해당 상병코드의 연령지역진료과 통계값 가져오기
    age_region_features = {
        '평균확률': row.get('평균확률', 0),
        '종합신뢰도': row.get('종합신뢰도', 0),
        '진료과_지역_상호작용': row.get('진료과_지역_상호작용', 0),
        '총샘플수_log': row.get('총샘플수_log', 0)
    }
    
    try:
        bin_label, est_cost = predict_cost_bin(
            row['상병코드'], row['지역'],
            lgb_clf, feat_cols, bins, age_region_features
        )
        preds.append((bin_label, est_cost))
    except:
        preds.append((0, 0))

# 예측 결과를 df에 컬럼으로 추가
df['pred_bin_age_region'], df['pred_cost_age_region'] = zip(*preds)

# CSV로 저장
output_path = f"{results_dir}/진료비_구간예측결과_연령지역진료과연계.csv"
df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"예측 결과를 '{output_path}'에 저장했습니다.")
print(f"모든 결과가 '{results_dir}' 디렉토리에 저장되었습니다!") 