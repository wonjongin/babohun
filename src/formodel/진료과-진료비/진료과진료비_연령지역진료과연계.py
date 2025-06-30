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
    mean_absolute_error, mean_squared_error, r2_score
)

# 모델 성능 저장용 리스트
model_performance = []

def calculate_classification_metrics(y_true, y_pred, model_name):
    """분류 모델 성능 지표 계산"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    return {
        'model_name': model_name,
        'model_type': 'classification',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def calculate_regression_metrics(y_true, y_pred, model_name):
    """회귀 모델 성능 지표 계산"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    return {
        'model_name': model_name,
        'model_type': 'regression',
        'mae': mae,
        'rmse': rmse,
        'r2_score': r2
    } 

# ----------------------------------------------------------------------
# 1) 데이터 로드 & 전처리
# ----------------------------------------------------------------------
print("=== 데이터 로딩 시작 ===")

data_csv = "final_merged_data/다빈도 질환 환자 연령별 분포.csv"
mapping_csv = "new_merged_data/df_result2_with_심평원.csv"

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

# 중복 확인 및 제거
print(f"원본 데이터 행 수: {len(df)}")
print(f"중복 행 수: {df.duplicated().sum()}")

# 중복이 너무 많으면 원본 데이터의 고유한 조합만 유지
if df.duplicated().sum() > len(df) * 0.5:  # 50% 이상 중복이면
    print("중복이 너무 많아 원본 데이터의 고유한 조합만 유지합니다.")
    # 실제 데이터 구조에 맞는 컬럼 사용
    available_keys = ['상병코드', '지역', '구분', '연인원', '진료비(천원)', '진료과']
    # 존재하는 컬럼만 필터링
    existing_keys = [key for key in available_keys if key in df.columns]
    print(f"사용할 컬럼: {existing_keys}")
    df = df.drop_duplicates(subset=existing_keys)
    print(f"고유 조합 기준 중복 제거 후 행 수: {len(df)}")
else:
    if df.duplicated().sum() > 0:
        print("일반적인 중복 제거를 수행합니다.")
        df = df.drop_duplicates()
        print(f"중복 제거 후 행 수: {len(df)}")

mapping = pd.read_csv(mapping_csv, encoding="utf-8-sig")
df = df.merge(mapping[['상병코드', '진료과']], on='상병코드', how='left')
df.dropna(subset=['진료과'], inplace=True)
print(f"진료과 매핑 후 행 수: {len(df)}")

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
    print(f"진료과 통계 merge 전 행 수: {len(df)}")
    print(f"진료과 통계 데이터 진료과 수: {len(department_stats)}")
    
    # merge 전 중복 확인
    print(f"merge 전 중복 행 수: {df.duplicated().sum()}")
    
    df = df.merge(department_stats, on='진료과', how='left')
    
    # merge 후 중복 확인
    print(f"진료과 통계 merge 후 행 수: {len(df)}")
    print(f"merge 후 중복 행 수: {df.duplicated().sum()}")
    
    # 중복이 너무 많으면 원본 데이터의 고유한 조합만 유지
    if df.duplicated().sum() > len(df) * 0.5:  # 50% 이상 중복이면
        print("중복이 너무 많아 원본 데이터의 고유한 조합만 유지합니다.")
        # 원본 데이터의 고유한 조합 기준으로 중복 제거
        original_keys = ['상병코드', '지역', '연령대', '구분', '연인원', '진료비', '진료과']
        df = df.drop_duplicates(subset=original_keys)
        print(f"고유 조합 기준 중복 제거 후 행 수: {len(df)}")
    else:
        if df.duplicated().sum() > 0:
            print("일반적인 중복 제거를 수행합니다.")
            df = df.drop_duplicates()
            print(f"중복 제거 후 행 수: {len(df)}")
    
    # 지역별 통계 정보 추출
    region_stats = df_age_region.groupby('지역').agg({
        'top1_probability': 'mean',
        'confidence': 'mean',
        'sample_weight': 'sum',
        'age_num': 'mean'
    }).reset_index()
    
    region_stats.columns = ['지역', '지역평균확률', '지역평균신뢰도', '지역총샘플수', '지역평균연령']
    
    # 지역별로 매핑
    print(f"지역 통계 merge 전 행 수: {len(df)}")
    print(f"지역 통계 데이터 지역 수: {len(region_stats)}")
    
    # merge 전 중복 확인
    print(f"merge 전 중복 행 수: {df.duplicated().sum()}")
    
    df = df.merge(region_stats, on='지역', how='left')
    
    # merge 후 중복 확인
    print(f"지역 통계 merge 후 행 수: {len(df)}")
    print(f"merge 후 중복 행 수: {df.duplicated().sum()}")
    
    # 중복이 너무 많으면 원본 데이터의 고유한 조합만 유지
    if df.duplicated().sum() > len(df) * 0.5:  # 50% 이상 중복이면
        print("중복이 너무 많아 원본 데이터의 고유한 조합만 유지합니다.")
        # 원본 데이터의 고유한 조합 기준으로 중복 제거
        original_keys = ['상병코드', '지역', '연령대', '구분', '연인원', '진료비', '진료과']
        df = df.drop_duplicates(subset=original_keys)
        print(f"고유 조합 기준 중복 제거 후 행 수: {len(df)}")
    else:
        if df.duplicated().sum() > 0:
            print("일반적인 중복 제거를 수행합니다.")
            df = df.drop_duplicates()
            print(f"중복 제거 후 행 수: {len(df)}")
    
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
groups = [g['진료비(천원)'].values for _, g in df.groupby('상병코드') if len(g) >= 3]
H, p = stats.kruskal(*groups)
print(f"=== Kruskal–Wallis 검정: H={H:.4f}, p-value={p:.4e} ===")

dunn = sp.posthoc_dunn(df, val_col='진료비(천원)', group_col='상병코드', p_adjust='bonferroni')
print("=== Dunn's post-hoc (Bonferroni) ===")
print(dunn)

# ----------------------------------------------------------------------
# 4) 분류 모델: 고비용 여부 예측 (연령지역진료과 데이터 포함)
# ----------------------------------------------------------------------
thr = df['진료비(천원)'].quantile(0.75)
df['high_cost'] = (df['진료비(천원)'] >= thr).astype(int)

# Decision Tree (연령지역진료과 데이터 포함)
X_dt = pd.get_dummies(df[['상병코드', '평균확률', '종합신뢰도']], prefix='', prefix_sep='')
y = df['high_cost']
X_tr_dt, X_te_dt, y_tr, y_te = train_test_split(
    X_dt, y, test_size=0.3, random_state=42, stratify=y
)
dt = DecisionTreeClassifier(max_depth=4, class_weight='balanced', random_state=42)
dt.fit(X_tr_dt, y_tr)
y_pred_dt = dt.predict(X_te_dt)
print("\n=== DecisionTreeClassifier (연령지역진료과 데이터 포함) ===")
print(classification_report(y_te, y_pred_dt))

# 성능 저장
dt_performance = calculate_classification_metrics(y_te, y_pred_dt, "DecisionTree_Classification")
model_performance.append(dt_performance)

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
y_pred_rf = rf.predict(X_te_rf)
y_pred_gb = gb.predict(X_te_rf)
print("\n=== RandomForestClassifier (연령지역진료과 데이터 포함) ===")
print(classification_report(y_te, y_pred_rf))
print("\n=== GradientBoostingClassifier (연령지역진료과 데이터 포함) ===")
print(classification_report(y_te, y_pred_gb))

# 성능 저장
rf_performance = calculate_classification_metrics(y_te, y_pred_rf, "RandomForest_Classification")
gb_performance = calculate_classification_metrics(y_te, y_pred_gb, "GradientBoosting_Classification")
model_performance.extend([rf_performance, gb_performance])

# ----------------------------------------------------------------------
# 5) 회귀 모델: 진료비 직접 예측 (연령지역진료과 데이터 포함)
# ----------------------------------------------------------------------
X_reg = X_rf.copy()
y_reg = df['진료비(천원)'].values
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
    
    # 성능 저장
    reg_performance = calculate_regression_metrics(y_te_rg, pred, f"{name}_Regression")
    model_performance.append(reg_performance)

# ----------------------------------------------------------------------
# 6) 로그 스케일 기반 진료비 구간 예측 (연령지역진료과 데이터 포함)
# ----------------------------------------------------------------------
# 6.1) 로그 스케일 구간 정의
min_v = df['진료비(천원)'].min()
max_v = df['진료비(천원)'].max()
bins = np.logspace(np.log10(min_v), np.log10(max_v), num=6)
# 6.2) 구간 클래스 할당
labels = pd.cut(df['진료비(천원)'], bins=bins, labels=False, include_lowest=True)
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

# LightGBM 성능 저장
lgb_performance = calculate_classification_metrics(y_te, y_pred, "LightGBM_Classification")
model_performance.append(lgb_performance)

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

# 최종 중복 확인 및 제거
print(f"예측 완료 후 데이터 행 수: {len(df)}")
print(f"최종 중복 행 수: {df.duplicated().sum()}")
if df.duplicated().sum() > 0:
    print("최종 중복 행을 제거합니다.")
    df = df.drop_duplicates()
    print(f"최종 중복 제거 후 행 수: {len(df)}")

# CSV로 저장
output_path = f"{results_dir}/진료비_구간예측결과_연령지역진료과연계.csv"
df.to_csv(output_path, index=False, encoding='utf-8-sig')

# 모델 성능 요약 및 CSV 저장
print("\n" + "="*60)
print("모델별 성능 요약")
print("="*60)

# 분류 모델 성능 요약
classification_models = [p for p in model_performance if p['model_type'] == 'classification']
if classification_models:
    print("\n📊 분류 모델 성능:")
    print("-" * 50)
    for model in classification_models:
        print(f"{model['model_name']}:")
        print(f"  정확도: {model['accuracy']:.4f}")
        print(f"  정밀도: {model['precision']:.4f}")
        print(f"  재현율: {model['recall']:.4f}")
        print(f"  F1점수: {model['f1_score']:.4f}")
        print()

# 회귀 모델 성능 요약
regression_models = [p for p in model_performance if p['model_type'] == 'regression']
if regression_models:
    print("\n📈 회귀 모델 성능:")
    print("-" * 50)
    for model in regression_models:
        print(f"{model['model_name']}:")
        print(f"  MAE: {model['mae']:.0f}천원")
        print(f"  RMSE: {model['rmse']:.0f}천원")
        print(f"  R²: {model['r2_score']:.4f}")
        print()

# 성능 데이터를 DataFrame으로 변환하여 CSV 저장
performance_df = pd.DataFrame(model_performance)
performance_csv_path = f"{results_dir}/모델별_성능_요약.csv"
performance_df.to_csv(performance_csv_path, index=False, encoding='utf-8-sig')

print(f"모델 성능 요약이 '{performance_csv_path}'에 저장되었습니다.")
print(f"예측 결과를 '{output_path}'에 저장했습니다.")
print(f"모든 결과가 '{results_dir}' 디렉토리에 저장되었습니다!")