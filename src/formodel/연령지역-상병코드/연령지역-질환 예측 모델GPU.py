import pandas as pd
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
import subprocess
import os
import multiprocessing
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# CUDA 환경 변수 설정 (GPU 사용 강제)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XGBOOST_USE_CUDA'] = '1'
os.environ['LIGHTGBM_USE_GPU'] = '1'

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    top_k_accuracy_score, balanced_accuracy_score, roc_auc_score
)
from sklearn.preprocessing import label_binarize

from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTEENN

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# 특정 경고만 무시
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.preprocessing._encoders")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.feature_selection._univariate_selection")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.feature_selection._univariate_selection")
warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn.linear_model._sag")

# GPU 사용 가능 여부 확인
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ GPU 사용 가능: NVIDIA GPU가 감지되었습니다.")
        print("🚀 GPU 가속이 활성화되어 학습 속도가 크게 향상됩니다.")
        
        # CUDA 버전 확인
        try:
            cuda_result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
            if cuda_result.returncode == 0:
                cuda_version = cuda_result.stdout.split('release ')[1].split(',')[0]
                print(f"🔧 CUDA 버전: {cuda_version}")
        except:
            print("⚠️ CUDA 버전 확인 실패")
        
        # GPU 상세 정보 출력
        gpu_info = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], 
                                capture_output=True, text=True)
        if gpu_info.returncode == 0:
            print(f"📊 GPU 정보: {gpu_info.stdout.strip()}")
        
        # 초기 GPU 사용률 확인
        gpu_util = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                                capture_output=True, text=True)
        if gpu_util.returncode == 0:
            print(f"🖥️ 초기 GPU 사용률: {gpu_util.stdout.strip()}%")
            
        # 환경 변수 확인
        print(f"🔧 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
        print(f"🔧 XGBOOST_USE_CUDA: {os.environ.get('XGBOOST_USE_CUDA', 'Not set')}")
        print(f"🔧 LIGHTGBM_USE_GPU: {os.environ.get('LIGHTGBM_USE_GPU', 'Not set')}")
            
    else:
        print("⚠️ GPU 사용 불가: NVIDIA GPU가 감지되지 않았습니다.")
        print("CPU 모드로 실행됩니다.")
except:
    print("⚠️ GPU 사용 불가: nvidia-smi 명령어를 찾을 수 없습니다.")
    print("CPU 모드로 실행됩니다.")

print()

# CPU 코어 수 확인
cpu_count = multiprocessing.cpu_count()
print(f"🔧 시스템 정보:")
print(f"  - CPU 코어 수: {cpu_count}개")
print(f"  - GPU 사용: 활성화")
print()

# --------------------------------------------------
# 1) 데이터 적재 및 가공
# --------------------------------------------------
df = pd.read_csv("new_merged_data/df_result2_mapping1.csv", dtype=str)
age_cols = ["59이하", "60-64", "65-69", "70-79", "80-89", "90이상"]

m = df.melt(
    id_vars=["년도", "구분", "지역", "상병코드", "진료과"],
    value_vars=age_cols,
    var_name="age_group",
    value_name="count",
)
m["count"] = pd.to_numeric(m["count"], errors="coerce").fillna(0).astype(int)
m["대표진료과"] = m["진료과"]
train = m[m["대표진료과"].notna()]

# 강화된 피처 엔지니어링
train["year_num"] = train["년도"].astype(int) - train["년도"].astype(int).min()

# 연령대 수치화 (중간값 사용)
age_mapping = {
    "59이하": 30, "60-64": 62, "65-69": 67, 
    "70-79": 75, "80-89": 85, "90이상": 95
}
train["age_num"] = train["age_group"].map(age_mapping)

# 지역별 특성 (대도시 vs 중소도시)
major_cities = ["서울", "부산", "대구", "인천", "광주", "대전"]
train["is_major_city"] = train["지역"].isin(major_cities).astype(int)

# 구분별 특성 (입원 vs 외래)
train["is_inpatient"] = (train["구분"] == "입원").astype(int)

# 상병코드 기반 피처 (첫 3자리로 그룹화)
train["disease_group"] = train["상병코드"].str[:3]

# 연도별 트렌드
train["year_trend"] = train["year_num"] ** 2

# 복합 피처
train["age_city_interaction"] = train["age_num"] * train["is_major_city"]
train["age_year_interaction"] = train["age_num"] * train["year_num"]

# 지역-연령대 조합
train["region_age"] = train["지역"] + "_" + train["age_group"]

X = train[["year_num", "age_num", "is_major_city", "is_inpatient", 
           "year_trend", "age_city_interaction", "age_year_interaction", 
           "지역", "age_group", "구분", "disease_group", "region_age"]]
y = train["대표진료과"]
w = train["count"]

# --------------------------------------------------
# 2) 학습 / 검증 분리
# --------------------------------------------------
X_tr, X_te, y_tr, y_te, w_tr, w_te = train_test_split(
    X, y, w, test_size=0.20, stratify=y, random_state=42
)

# --------------------------------------------------
# 3) 전처리 파이프라인
# --------------------------------------------------
# 수치형 피처
num_cols = ["year_num", "age_num", "is_major_city", "is_inpatient", 
            "year_trend", "age_city_interaction", "age_year_interaction"]

# 범주형 피처
cat_cols = ["지역", "age_group", "구분", "disease_group", "region_age"]

preprocessor = ColumnTransformer(
    [
        ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_cols),
        ("scale", StandardScaler(), num_cols),
    ]
)

# --------------------------------------------------
# 4) XGB용 래퍼 클래스
# --------------------------------------------------
class XGBWrapper(XGBClassifier):
    """문자→숫자 라벨을 내부 변환하고, 원본 라벨은 orig_classes_에 저장"""
    def fit(self, X, y, **kwargs):
        self._le = LabelEncoder()
        y_enc = self._le.fit_transform(y)
        super().fit(X, y_enc, **kwargs)
        self.orig_classes_ = self._le.classes_
        return self

    def predict(self, X):
        return self._le.inverse_transform(super().predict(X))

    def predict_proba(self, X):
        return super().predict_proba(X)

# --------------------------------------------------
# 5) 파이프라인 & 그리드 정의 함수
# --------------------------------------------------
def make_pipeline(clf, param_grid):
    pipe = ImbPipeline(
        [
            ("prep", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("variance", VarianceThreshold(threshold=0.01)),  # 상수 피처 제거
            ("select", SelectKBest(f_classif)),
            ("clf", clf),
        ]
    )
    return pipe, param_grid

# k 값을 동적으로 설정
# preprocessor를 먼저 fit시켜야 함
preprocessor.fit(X_tr)
n_features_after_prep = len(preprocessor.get_feature_names_out())  # 전처리 후 피처 수
max_k = min(n_features_after_prep, 100)  # 최대 100개로 확장

# Logistic Regression
pipe_lr, params_lr = make_pipeline(
    LogisticRegression(
        penalty="l1", solver="saga", max_iter=5000, class_weight="balanced"
    ),
    {
        "select__k": [max_k//4, max_k//2, max_k],
        "clf__C": [0.001, 0.01, 0.1, 1, 10],
    },
)

# Random Forest
pipe_rf, params_rf = make_pipeline(
    RandomForestClassifier(class_weight="balanced", random_state=42),
    {
        "select__k": [max_k//4, max_k//2, max_k],
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [None, 10, 20, 30],
        "clf__min_samples_split": [2, 5, 10],
    },
)

# XGBoost
pipe_xgb, params_xgb = make_pipeline(
    XGBWrapper(
        eval_metric="mlogloss",
        random_state=42,
        tree_method="hist",  # hist 사용
        device="cuda",  # 새로운 GPU 설정 방식
        max_bin=256,  # GPU 메모리 최적화
        single_precision_histogram=True,  # GPU 메모리 절약
        enable_categorical=False,  # 카테고리형 비활성화
        max_leaves=0,  # GPU 최적화
        grow_policy="lossguide",  # GPU 최적화
    ),
    {
        "select__k": [max_k//4, max_k//2, max_k],
        "clf__n_estimators": [200, 400, 600],
        "clf__max_depth": [3, 6, 9],
        "clf__learning_rate": [0.01, 0.1, 0.2],
        "clf__reg_alpha": [0, 0.1, 1],
        "clf__reg_lambda": [0, 0.1, 1],
    },
)

# LightGBM
pipe_lgb, params_lgb = make_pipeline(
    LGBMClassifier(
        objective="multiclass",
        random_state=42,
        class_weight="balanced",
        verbose=-1,
        device="gpu",  # GPU 사용
        gpu_platform_id=0,  # GPU 플랫폼 ID
        gpu_device_id=0,  # GPU 디바이스 ID
        force_col_wise=True,  # GPU 최적화
        gpu_use_dp=False,  # 단정밀도 사용으로 메모리 절약
        max_bin=255,  # GPU 최적화
        num_leaves=31,  # 고정값으로 경고 제거
        min_child_samples=20,  # 고정값으로 경고 제거
        subsample=1.0,  # 고정값으로 경고 제거
        colsample_bytree=1.0,  # 고정값으로 경고 제거
        # 성능 개선을 위한 추가 설정
        n_jobs=1,  # GPU 사용시 단일 스레드
        deterministic=True,  # 재현성 보장
        force_row_wise=False,  # GPU 최적화
        # GPU 강제 사용을 위한 추가 설정
        gpu_use_dp_for_histogram=False,  # 히스토그램도 단정밀도
        gpu_use_dp_for_histogram_bin=False,  # 히스토그램 빈도도 단정밀도
        gpu_use_dp_for_histogram_bin_leaf=False,  # 리프별 히스토그램도 단정밀도
        gpu_use_dp_for_histogram_bin_leaf_grad=False,  # 그래디언트도 단정밀도
        gpu_use_dp_for_histogram_bin_leaf_hess=False,  # 헤시안도 단정밀도
        gpu_use_dp_for_histogram_bin_leaf_hess_grad=False,  # 헤시안 그래디언트도 단정밀도
        gpu_use_dp_for_histogram_bin_leaf_hess_grad_hess=False,  # 헤시안 그래디언트 헤시안도 단정밀도
    ),
    {
        "select__k": [max_k//4, max_k//2, max_k],
        "clf__n_estimators": [200, 400, 600],
        "clf__max_depth": [3, 6, 9],
        "clf__learning_rate": [0.01, 0.1, 0.2],
        "clf__num_leaves": [31, 63, 127],
    },
)

# Gradient Boosting
pipe_gb, params_gb = make_pipeline(
    GradientBoostingClassifier(random_state=42),
    {
        "select__k": [max_k//4, max_k//2, max_k],
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [3, 6, 9],
        "clf__learning_rate": [0.01, 0.1, 0.2],
    },
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --------------------------------------------------
# 6) 그리드 서치 실행
# --------------------------------------------------
print("=== 모델 학습 시작 ===")
print(f"전처리 후 피처 수: {n_features_after_prep}")
print(f"최대 k 값: {max_k}")
print(f"학습 데이터 크기: {X_tr.shape}")
print(f"테스트 데이터 크기: {X_te.shape}")
print(f"클래스 수: {len(np.unique(y_tr))}")
print(f"클래스별 샘플 수:")
for cls, count in y_tr.value_counts().items():
    print(f"  {cls}: {count}개")
print()

grids = {}
for i, (name, (pipe, params)) in enumerate(zip(
    ["lr", "rf", "xgb", "lgb", "gb"],
    [
        (pipe_lr, params_lr),
        (pipe_rf, params_rf),
        (pipe_xgb, params_xgb),
        (pipe_lgb, params_lgb),
        (pipe_gb, params_gb),
    ],
), 1):
    print(f"=== 모델 {i}/5: {name.upper()} 모델 학습 중... ===")
    print(f"파라미터 조합 수: {len([(k, v) for k, v in params.items() for v in v])}")
    print(f"예상 학습 시간: 약 1-3분")
    
    # GPU 사용 확인 (XGBoost, LightGBM의 경우)
    if name in ['xgb', 'lgb']:
        print(f"🔍 {name.upper()} GPU 사용 확인 중...")
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            print(f"nvidia-smi 명령어 결과: {result.returncode}")
            print(f"nvidia-smi 출력: {result.stdout.strip()}")
            if result.returncode == 0:
                gpu_util = result.stdout.strip()
                print(f"🖥️ 학습 전 GPU 사용률: {gpu_util}%")
            else:
                print(f"❌ nvidia-smi 오류: {result.stderr}")
        except Exception as e:
            print(f"❌ GPU 확인 중 오류: {str(e)}")
        
        # GPU 메모리 사용량도 확인
        try:
            mem_result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True)
            if mem_result.returncode == 0:
                mem_info = mem_result.stdout.strip().split(',')
                if len(mem_info) >= 2:
                    print(f"💾 GPU 메모리: {mem_info[0]}/{mem_info[1]} MB")
        except:
            pass
    
    # CPU 모델들: 멀티코어 활용
    n_jobs = max(1, int(cpu_count * 0.75))  # 75% 코어 활용
    
    grid = GridSearchCV(
        pipe, params, cv=cv, scoring="accuracy", n_jobs=n_jobs, verbose=1
    )
    print(f"GridSearchCV 시작...")
    grid.fit(X_tr, y_tr)  # sample_weight 미사용
    grids[name] = grid
    
    # 학습 후 GPU 사용률 확인
    if name in ['xgb', 'lgb']:
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                gpu_util = result.stdout.strip()
                print(f"🖥️ 학습 후 GPU 사용률: {gpu_util}%")
        except:
            pass
    
    print(f"✅ {name.upper()} 최적 파라미터: {grid.best_params_}")
    print(f"✅ {name.upper()} 최적 점수: {grid.best_score_:.4f}")
    print(f"✅ {name.upper()} 학습 완료 ({i}/5)")
    print()

# --------------------------------------------------
# 7) 앙상블 (Voting & Stacking)
# --------------------------------------------------
print("=== 앙상블 모델 학습 중... ===")

# estimators 정의
estimators = [(n, grids[n].best_estimator_) for n in ["lr", "rf", "xgb", "lgb", "gb"]]
print(f"앙상블에 사용할 모델: {[name for name, _ in estimators]}")

print("1/2: Voting Classifier 학습 중...")
vot = VotingClassifier(estimators=estimators, voting="soft")
vot.fit(X_tr, y_tr)
print("✅ Voting Classifier 완료")

print("2/2: Stacking Classifier 학습 중...")
print("  - 메타 모델: LogisticRegression")
print("  - 교차검증: 5-fold")
print("  - 병렬 처리: CPU 모델과 GPU 모델 혼재로 인해 단일 스레드 사용")

# Stacking: 병렬 처리
n_jobs = max(1, int(cpu_count * 0.5))   # 50% 코어 활용
stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=5000),
    cv=cv,
    n_jobs=n_jobs,
)
stack.fit(X_tr, y_tr)
print("✅ Stacking Classifier 완료")
print()

# --------------------------------------------------
# 8) 평가 함수
# --------------------------------------------------
def eval_model(name, model, X, y_true, w):
    print(f"=== {name} 모델 평가 중... ===")
    y_pred = model.predict(X)
    proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None

    print(f"\n=== {name} ===")
    acc = accuracy_score(y_true, y_pred, sample_weight=w)
    macro_f1 = f1_score(y_true, y_pred, average="macro", sample_weight=w)
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")

    if proba is not None:
        # ---- class_order 안전 추출 ----
        class_order = getattr(model, "orig_classes_", None)
        if class_order is None:
            class_order = getattr(model, "classes_", None)
        if class_order is None:
            class_order = np.unique(y_true)
        # --------------------------------

        top3_acc = top_k_accuracy_score(y_true, proba, k=3, sample_weight=w)
        bal_acc = balanced_accuracy_score(y_true, y_pred, sample_weight=w)
        print(f"Top-3 Accuracy: {top3_acc:.4f}")
        print(f"Balanced Accuracy: {bal_acc:.4f}")

        y_bin = label_binarize(y_true, classes=class_order)
        roc_auc = roc_auc_score(y_bin, proba, average="macro", sample_weight=w)
        print(f"Macro ROC-AUC: {roc_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, sample_weight=w, digits=3))

print("=== 기본 모델 평가 시작 ===")
print("평가할 모델: XGB, LGB, GB, Voting, Stacking")
print()

for i, (nm, mdl) in enumerate([
    ("XGB", grids["xgb"].best_estimator_),
    ("LGB", grids["lgb"].best_estimator_),
    ("GB", grids["gb"].best_estimator_),
    ("Voting", vot),
    ("Stacking", stack),
], 1):
    print(f"--- 모델 {i}/5: {nm} ---")
    eval_model(nm, mdl, X_te, y_te, w_te)
    print()

print("\n" + "="*60)
print("=== 개선된 샘플링 기법 테스트 시작 ===")
print("="*60)

# 1. 극단적 가중치 계산
print("1/3: 극단적 가중치 계산 중...")
# 문자열 라벨을 정수로 변환
le_weights = LabelEncoder()
y_tr_encoded = le_weights.fit_transform(y_tr)
class_counts = np.bincount(y_tr_encoded)
total_samples = len(y_tr_encoded)
extreme_weights = (total_samples / (len(class_counts) * class_counts)) ** 1.5
print(f"가중치 범위: {extreme_weights.min():.2f} ~ {extreme_weights.max():.2f}")
print("✅ 가중치 계산 완료")
print()

# 2. 다양한 샘플링 기법 시도
sampling_methods = {
    'adasyn': ADASYN(random_state=42),
    'borderline_smote': BorderlineSMOTE(random_state=42),
    'smote_enn': SMOTEENN(random_state=42)
}

print(f"2/3: {len(sampling_methods)}가지 샘플링 기법 테스트")
print(f"테스트할 기법: {list(sampling_methods.keys())}")
print()

# 3. 각 방법별로 모델 학습 및 평가
for i, (name, sampler) in enumerate(sampling_methods.items(), 1):
    print(f"--- 샘플링 기법 {i}/{len(sampling_methods)}: {name.upper()} ---")
    print(f"샘플러: {type(sampler).__name__}")
    print(f"예상 처리 시간: 약 30초-1분")
    
    try:
        # 전처리 적용
        print("전처리 적용 중...")
        X_tr_preprocessed = preprocessor.transform(X_tr)
        print(f"전처리 후 데이터 크기: {X_tr_preprocessed.shape}")
        
        # 샘플링 적용 (전처리된 데이터 사용)
        print("샘플링 적용 중...")
        X_resampled, y_resampled = sampler.fit_resample(X_tr_preprocessed, y_tr)
        print(f"샘플링 후 데이터 크기: {X_resampled.shape}")
        print(f"샘플링 후 클래스별 샘플 수:")
        for cls, count in pd.Series(y_resampled).value_counts().items():
            print(f"  {cls}: {count}개")
        
        # 샘플링 후 새로운 가중치 계산
        print("샘플링 후 가중치 계산 중...")
        le_resampled = LabelEncoder()
        y_resampled_encoded = le_resampled.fit_transform(y_resampled)
        class_counts_resampled = np.bincount(y_resampled_encoded)
        total_samples_resampled = len(y_resampled_encoded)
        resampled_weights = (total_samples_resampled / (len(class_counts_resampled) * class_counts_resampled)) ** 1.5
        sample_weights_resampled = np.array([resampled_weights[label] for label in y_resampled_encoded])
        print(f"샘플링 후 가중치 범위: {resampled_weights.min():.2f} ~ {resampled_weights.max():.2f}")
        
        # XGBWrapper 사용하여 문자열 라벨 처리
        pipe = Pipeline([
            ('clf', XGBWrapper(
                eval_metric="mlogloss",
                random_state=42,
                tree_method="hist",  # hist 사용
                device="cuda",  # 새로운 GPU 설정 방식
                max_bin=256,  # GPU 메모리 최적화
                single_precision_histogram=True,  # GPU 메모리 절약
                enable_categorical=False,  # 카테고리형 비활성화
                max_leaves=0,  # GPU 최적화
                grow_policy="lossguide",  # GPU 최적화
            ))
        ])
        
        print("모델 학습 중...")
        # GPU 사용률 확인
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                gpu_util = result.stdout.strip()
                print(f"🖥️ 샘플링 학습 전 GPU 사용률: {gpu_util}%")
        except:
            pass
            
        pipe.fit(X_resampled, y_resampled, clf__sample_weight=sample_weights_resampled)
        
        # 학습 후 GPU 사용률 확인
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                gpu_util = result.stdout.strip()
                print(f"🖥️ 샘플링 학습 후 GPU 사용률: {gpu_util}%")
        except:
            pass
            
        print(f"✅ {name.upper()} 학습 완료")

        # 평가 (테스트 데이터도 전처리)
        print(f"{name.upper()} 평가 중...")
        X_te_preprocessed = preprocessor.transform(X_te)
        y_pred = pipe.predict(X_te_preprocessed)
        eval_model(f"{name.upper()}", pipe, X_te_preprocessed, y_te, w_te)
        
    except Exception as e:
        print(f"❌ {name.upper()} 처리 중 오류 발생: {str(e)}")
        print("다음 샘플링 기법으로 진행...")
    
    print()

print("3/3: 모든 모델 학습 및 평가 완료")
print("="*60)
print("🎉 모든 분석이 완료되었습니다!")
print("="*60)

# --------------------------------------------------
# 9) 데이터 추출 및 저장
# --------------------------------------------------
print("\n" + "="*60)
print("=== 데이터 추출 및 저장 시작 ===")
print("="*60)

# 결과 저장 디렉토리 생성
results_dir = "model_results"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(f"{results_dir}/predictions", exist_ok=True)
os.makedirs(f"{results_dir}/performance", exist_ok=True)
os.makedirs(f"{results_dir}/features", exist_ok=True)

print(f"📁 결과 저장 디렉토리: {results_dir}/")

# --------------------------------------------------
# 1) 모델 성능 비교 결과 저장
# --------------------------------------------------
print("1/6: 모델 성능 비교 결과 저장 중...")

def extract_model_performance(model_name, model, X, y_true, w):
    """모델 성능 지표 추출"""
    y_pred = model.predict(X)
    proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    
    # 기본 성능 지표
    acc = accuracy_score(y_true, y_pred, sample_weight=w)
    macro_f1 = f1_score(y_true, y_pred, average="macro", sample_weight=w)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", sample_weight=w)
    bal_acc = balanced_accuracy_score(y_true, y_pred, sample_weight=w)
    
    # 클래스별 F1-score
    f1_scores = f1_score(y_true, y_pred, average=None, sample_weight=w)
    
    # Top-k 정확도
    top3_acc = top_k_accuracy_score(y_true, proba, k=3, sample_weight=w) if proba is not None else None
    top5_acc = top_k_accuracy_score(y_true, proba, k=5, sample_weight=w) if proba is not None else None
    
    # ROC-AUC
    if proba is not None:
        class_order = getattr(model, "orig_classes_", None)
        if class_order is None:
            class_order = getattr(model, "classes_", None)
        if class_order is None:
            class_order = np.unique(y_true)
        
        y_bin = label_binarize(y_true, classes=class_order)
        roc_auc_macro = roc_auc_score(y_bin, proba, average="macro", sample_weight=w)
        roc_auc_weighted = roc_auc_score(y_bin, proba, average="weighted", sample_weight=w)
    else:
        roc_auc_macro = None
        roc_auc_weighted = None
    
    return {
        "model_name": model_name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": bal_acc,
        "top3_accuracy": top3_acc,
        "top5_accuracy": top5_acc,
        "roc_auc_macro": roc_auc_macro,
        "roc_auc_weighted": roc_auc_weighted,
        "f1_scores_by_class": f1_scores.tolist() if f1_scores is not None else None,
        "classes": class_order.tolist() if 'class_order' in locals() else None
    }

# 기본 모델들 성능 추출
basic_models = {
    "XGB": grids["xgb"].best_estimator_,
    "LGB": grids["lgb"].best_estimator_,
    "GB": grids["gb"].best_estimator_,
    "RF": grids["rf"].best_estimator_,
    "LR": grids["lr"].best_estimator_,
    "Voting": vot,
    "Stacking": stack
}

performance_results = []
for name, model in basic_models.items():
    print(f"  - {name} 모델 성능 추출 중...")
    perf = extract_model_performance(name, model, X_te, y_te, w_te)
    performance_results.append(perf)

# 샘플링 모델들 성능 추출
sampling_results = []
for name, sampler in sampling_methods.items():
    try:
        # 샘플링 모델 재학습 (간단한 버전)
        X_tr_preprocessed = preprocessor.transform(X_tr)
        X_resampled, y_resampled = sampler.fit_resample(X_tr_preprocessed, y_tr)
        
        # 간단한 XGBoost 모델로 성능 측정
        simple_model = XGBWrapper(
            eval_metric="mlogloss",
            random_state=42,
            tree_method="hist",
            device="cuda",
            n_estimators=100
        )
        simple_model.fit(X_resampled, y_resampled)
        
        print(f"  - {name.upper()} 샘플링 모델 성능 추출 중...")
        perf = extract_model_performance(f"{name.upper()}_sampling", simple_model, X_te, y_te, w_te)
        sampling_results.append(perf)
    except Exception as e:
        print(f"  - {name.upper()} 샘플링 모델 성능 추출 실패: {str(e)}")

# 성능 결과 저장
performance_df = pd.DataFrame(performance_results + sampling_results)
performance_df.to_csv(f"{results_dir}/performance/model_performance_comparison.csv", index=False, encoding='utf-8-sig')

# JSON 형태로도 저장 (메타데이터 포함)
performance_metadata = {
    "timestamp": datetime.now().isoformat(),
    "test_data_size": len(X_te),
    "train_data_size": len(X_tr),
    "num_classes": len(np.unique(y_te)),
    "class_distribution": y_te.value_counts().to_dict(),
    "models": performance_results + sampling_results
}

with open(f"{results_dir}/performance/performance_metadata.json", 'w', encoding='utf-8') as f:
    json.dump(performance_metadata, f, ensure_ascii=False, indent=2)

print(f"✅ 모델 성능 비교 결과 저장 완료: {results_dir}/performance/")

# --------------------------------------------------
# 2) 그리드 서치 결과 저장
# --------------------------------------------------
print("2/6: 그리드 서치 결과 저장 중...")

grid_search_results = {}
for name, grid in grids.items():
    print(f"  - {name.upper()} 그리드 서치 결과 추출 중...")
    
    # 최적 파라미터
    best_params = grid.best_params_
    
    # 모든 파라미터 조합의 결과
    cv_results = grid.cv_results_
    
    # 결과 요약
    grid_summary = {
        "model_name": name.upper(),
        "best_score": grid.best_score_,
        "best_params": best_params,
        "best_estimator_type": type(grid.best_estimator_).__name__,
        "cv_splits": cv_results.get('split0_test_score', []).shape[0] if 'split0_test_score' in cv_results else 5,
        "total_combinations": len(cv_results['mean_test_score']),
        "mean_test_scores": cv_results['mean_test_score'].tolist(),
        "std_test_scores": cv_results['std_test_score'].tolist(),
        "rank_test_scores": cv_results['rank_test_score'].tolist(),
        "param_combinations": []
    }
    
    # 각 파라미터 조합별 결과
    for i in range(len(cv_results['mean_test_score'])):
        param_combo = {}
        for param_name in cv_results['params'][i].keys():
            param_combo[param_name] = cv_results['params'][i][param_name]
        
        param_combo.update({
            "mean_test_score": cv_results['mean_test_score'][i],
            "std_test_score": cv_results['std_test_score'][i],
            "rank_test_score": cv_results['rank_test_scores'][i]
        })
        grid_summary["param_combinations"].append(param_combo)
    
    grid_search_results[name] = grid_summary

# 그리드 서치 결과 저장
with open(f"{results_dir}/performance/grid_search_results.json", 'w', encoding='utf-8') as f:
    json.dump(grid_search_results, f, ensure_ascii=False, indent=2)

# 요약 테이블 생성
grid_summary_df = pd.DataFrame([
    {
        "model": name,
        "best_score": result["best_score"],
        "best_params": str(result["best_params"]),
        "total_combinations": result["total_combinations"]
    }
    for name, result in grid_search_results.items()
])
grid_summary_df.to_csv(f"{results_dir}/performance/grid_search_summary.csv", index=False, encoding='utf-8-sig')

print(f"✅ 그리드 서치 결과 저장 완료: {results_dir}/performance/")

# --------------------------------------------------
# 3) 피처 중요도 저장
# --------------------------------------------------
print("3/6: 피처 중요도 저장 중...")

feature_importance_results = {}

# 피처 이름 가져오기
feature_names = preprocessor.get_feature_names_out()

for name, model in basic_models.items():
    print(f"  - {name} 피처 중요도 추출 중...")
    
    # 모델에서 피처 중요도 추출
    if hasattr(model, 'feature_importances_'):
        # 트리 기반 모델
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # 선형 모델
        importances = np.abs(model.coef_).mean(axis=0)
    else:
        # 앙상블 모델의 경우 개별 모델들의 중요도 평균
        if hasattr(model, 'estimators_'):
            importances = np.mean([est.feature_importances_ for est in model.estimators_ if hasattr(est, 'feature_importances_')], axis=0)
        else:
            importances = None
    
    if importances is not None:
        # 피처 중요도 데이터프레임 생성
        importance_df = pd.DataFrame({
            'feature_name': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        feature_importance_results[name] = {
            "model_name": name,
            "feature_importance": importance_df.to_dict('records'),
            "top_features": importance_df.head(20).to_dict('records')
        }
        
        # 개별 CSV 파일로 저장
        importance_df.to_csv(f"{results_dir}/features/{name}_feature_importance.csv", index=False, encoding='utf-8-sig')

# 전체 피처 중요도 요약
if feature_importance_results:
    # 모든 모델의 피처 중요도 평균
    all_importances = []
    for name, result in feature_importance_results.items():
        importance_df = pd.DataFrame(result["feature_importance"])
        importance_df['model'] = name
        all_importances.append(importance_df)
    
    if all_importances:
        combined_importance = pd.concat(all_importances, ignore_index=True)
        avg_importance = combined_importance.groupby('feature_name')['importance'].agg(['mean', 'std', 'count']).reset_index()
        avg_importance = avg_importance.sort_values('mean', ascending=False)
        avg_importance.to_csv(f"{results_dir}/features/average_feature_importance.csv", index=False, encoding='utf-8-sig')

# 피처 중요도 메타데이터 저장
with open(f"{results_dir}/features/feature_importance_metadata.json", 'w', encoding='utf-8') as f:
    json.dump(feature_importance_results, f, ensure_ascii=False, indent=2)

print(f"✅ 피처 중요도 저장 완료: {results_dir}/features/")

# --------------------------------------------------
# 4) 예측 확률 분포 저장
# --------------------------------------------------
print("4/6: 예측 확률 분포 저장 중...")

prediction_probability_results = {}

for name, model in basic_models.items():
    print(f"  - {name} 예측 확률 분포 추출 중...")
    
    try:
        # 예측 확률
        proba = model.predict_proba(X_te)
        
        # 클래스 이름 가져오기
        class_names = getattr(model, "orig_classes_", None)
        if class_names is None:
            class_names = getattr(model, "classes_", None)
        if class_names is None:
            class_names = [f"class_{i}" for i in range(proba.shape[1])]
        
        # 예측 확률 데이터프레임 생성
        proba_df = pd.DataFrame(proba, columns=class_names)
        proba_df['actual_class'] = y_te.values
        proba_df['predicted_class'] = model.predict(X_te)
        proba_df['confidence'] = proba.max(axis=1)  # 최대 확률값
        proba_df['model'] = name
        
        # Top-3 예측
        top3_indices = np.argsort(proba, axis=1)[:, -3:][:, ::-1]
        top3_classes = class_names[top3_indices]
        top3_probs = np.take_along_axis(proba, top3_indices, axis=1)
        
        proba_df['top1_class'] = top3_classes[:, 0]
        proba_df['top1_prob'] = top3_probs[:, 0]
        proba_df['top2_class'] = top3_classes[:, 1]
        proba_df['top2_prob'] = top3_probs[:, 1]
        proba_df['top3_class'] = top3_classes[:, 2]
        proba_df['top3_prob'] = top3_probs[:, 2]
        
        # 신뢰도 구간별 분석
        confidence_bins = [0, 0.5, 0.7, 0.8, 0.9, 1.0]
        confidence_labels = ['0-0.5', '0.5-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
        proba_df['confidence_bin'] = pd.cut(proba_df['confidence'], bins=confidence_bins, labels=confidence_labels)
        
        # 신뢰도 구간별 정확도
        confidence_accuracy = proba_df.groupby('confidence_bin').apply(
            lambda x: (x['actual_class'] == x['predicted_class']).mean()
        ).reset_index()
        confidence_accuracy.columns = ['confidence_bin', 'accuracy']
        
        prediction_probability_results[name] = {
            "model_name": name,
            "class_names": class_names.tolist(),
            "confidence_accuracy": confidence_accuracy.to_dict('records'),
            "probability_stats": {
                "mean_confidence": proba_df['confidence'].mean(),
                "std_confidence": proba_df['confidence'].std(),
                "min_confidence": proba_df['confidence'].min(),
                "max_confidence": proba_df['confidence'].max()
            }
        }
        
        # 개별 CSV 파일로 저장
        proba_df.to_csv(f"{results_dir}/predictions/{name}_prediction_probabilities.csv", index=False, encoding='utf-8-sig')
        
    except Exception as e:
        print(f"  - {name} 예측 확률 추출 실패: {str(e)}")

# 예측 확률 메타데이터 저장
with open(f"{results_dir}/predictions/prediction_probability_metadata.json", 'w', encoding='utf-8') as f:
    json.dump(prediction_probability_results, f, ensure_ascii=False, indent=2)

print(f"✅ 예측 확률 분포 저장 완료: {results_dir}/predictions/")

# --------------------------------------------------
# 5) 클래스별 성능 분석 저장
# --------------------------------------------------
print("5/6: 클래스별 성능 분석 저장 중...")

class_performance_results = {}

for name, model in basic_models.items():
    print(f"  - {name} 클래스별 성능 분석 중...")
    
    try:
        y_pred = model.predict(X_te)
        
        # 클래스 이름 가져오기
        class_names = getattr(model, "orig_classes_", None)
        if class_names is None:
            class_names = getattr(model, "classes_", None)
        if class_names is None:
            class_names = np.unique(y_te)
        
        # 클래스별 성능 지표
        from sklearn.metrics import precision_recall_fscore_support
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_te, y_pred, labels=class_names, average=None, sample_weight=w_te
        )
        
        # 클래스별 정확도 계산
        class_accuracy = {}
        for i, class_name in enumerate(class_names):
            class_mask = (y_te == class_name)
            if class_mask.sum() > 0:
                class_accuracy[class_name] = (y_pred[class_mask] == y_te[class_mask]).mean()
            else:
                class_accuracy[class_name] = 0.0
        
        # 클래스별 성능 데이터프레임
        class_perf_df = pd.DataFrame({
            'class_name': class_names,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': support,
            'accuracy': [class_accuracy.get(cls, 0.0) for cls in class_names]
        })
        
        # 혼동 행렬 계산
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_te, y_pred, labels=class_names, sample_weight=w_te)
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        
        class_performance_results[name] = {
            "model_name": name,
            "class_performance": class_perf_df.to_dict('records'),
            "confusion_matrix": cm_df.to_dict(),
            "overall_metrics": {
                "macro_precision": precision.mean(),
                "macro_recall": recall.mean(),
                "macro_f1": f1.mean(),
                "weighted_precision": np.average(precision, weights=support),
                "weighted_recall": np.average(recall, weights=support),
                "weighted_f1": np.average(f1, weights=support)
            }
        }
        
        # 개별 CSV 파일로 저장
        class_perf_df.to_csv(f"{results_dir}/performance/{name}_class_performance.csv", index=False, encoding='utf-8-sig')
        cm_df.to_csv(f"{results_dir}/performance/{name}_confusion_matrix.csv", encoding='utf-8-sig')
        
    except Exception as e:
        print(f"  - {name} 클래스별 성능 분석 실패: {str(e)}")

# 클래스별 성능 메타데이터 저장
with open(f"{results_dir}/performance/class_performance_metadata.json", 'w', encoding='utf-8') as f:
    json.dump(class_performance_results, f, ensure_ascii=False, indent=2)

print(f"✅ 클래스별 성능 분석 저장 완료: {results_dir}/performance/")

# --------------------------------------------------
# 6) 예측 결과 저장 (전체 데이터)
# --------------------------------------------------
print("6/6: 예측 결과 저장 중...")

# 테스트 데이터의 원본 피처들과 예측 결과를 결합
prediction_results = {}

for name, model in basic_models.items():
    print(f"  - {name} 예측 결과 저장 중...")
    
    try:
        # 예측
        y_pred = model.predict(X_te)
        proba = model.predict_proba(X_te) if hasattr(model, "predict_proba") else None
        
        # 클래스 이름 가져오기
        class_names = getattr(model, "orig_classes_", None)
        if class_names is None:
            class_names = getattr(model, "classes_", None)
        if class_names is None:
            class_names = [f"class_{i}" for i in range(proba.shape[1])] if proba is not None else []
        
        # 결과 데이터프레임 생성
        result_df = X_te.copy()
        result_df['actual_class'] = y_te.values
        result_df['predicted_class'] = y_pred
        result_df['prediction_correct'] = (y_te == y_pred)
        result_df['sample_weight'] = w_te.values
        result_df['model'] = name
        
        # 예측 확률 추가
        if proba is not None:
            for i, class_name in enumerate(class_names):
                result_df[f'prob_{class_name}'] = proba[:, i]
            result_df['confidence'] = proba.max(axis=1)
            
            # Top-3 예측
            top3_indices = np.argsort(proba, axis=1)[:, -3:][:, ::-1]
            top3_classes = class_names[top3_indices]
            top3_probs = np.take_along_axis(proba, top3_indices, axis=1)
            
            result_df['top1_class'] = top3_classes[:, 0]
            result_df['top1_prob'] = top3_probs[:, 0]
            result_df['top2_class'] = top3_classes[:, 1]
            result_df['top2_prob'] = top3_probs[:, 1]
            result_df['top3_class'] = top3_classes[:, 2]
            result_df['top3_prob'] = top3_probs[:, 2]
        
        # 원본 데이터의 추가 정보도 포함
        # 테스트 데이터의 인덱스를 사용해서 원본 데이터에서 추가 정보 가져오기
        test_indices = X_te.index
        original_data = train.loc[test_indices]
        
        # 원본 데이터의 추가 컬럼들
        for col in ['년도', '구분', '지역', '상병코드', '진료과', 'count']:
            if col in original_data.columns:
                result_df[f'original_{col}'] = original_data[col].values
        
        prediction_results[name] = {
            "model_name": name,
            "total_predictions": len(result_df),
            "correct_predictions": result_df['prediction_correct'].sum(),
            "accuracy": result_df['prediction_correct'].mean(),
            "class_names": class_names.tolist() if isinstance(class_names, np.ndarray) else class_names
        }
        
        # 개별 CSV 파일로 저장
        result_df.to_csv(f"{results_dir}/predictions/{name}_complete_predictions.csv", index=False, encoding='utf-8-sig')
        
    except Exception as e:
        print(f"  - {name} 예측 결과 저장 실패: {str(e)}")

# 전체 예측 결과 요약
all_predictions_summary = pd.DataFrame(prediction_results.values())
all_predictions_summary.to_csv(f"{results_dir}/predictions/all_models_prediction_summary.csv", index=False, encoding='utf-8-sig')

# 예측 결과 메타데이터 저장
with open(f"{results_dir}/predictions/prediction_results_metadata.json", 'w', encoding='utf-8') as f:
    json.dump(prediction_results, f, ensure_ascii=False, indent=2)

print(f"✅ 예측 결과 저장 완료: {results_dir}/predictions/")

# --------------------------------------------------
# 최종 요약
# --------------------------------------------------
print("\n" + "="*60)
print("=== 데이터 추출 완료 ===")
print("="*60)

print(f"📁 저장된 파일들:")
print(f"  📊 성능 비교: {results_dir}/performance/")
print(f"    - model_performance_comparison.csv")
print(f"    - performance_metadata.json")
print(f"    - grid_search_results.json")
print(f"    - grid_search_summary.csv")
print(f"    - [모델명]_class_performance.csv")
print(f"    - [모델명]_confusion_matrix.csv")
print(f"    - class_performance_metadata.json")
print()
print(f"  🔍 피처 중요도: {results_dir}/features/")
print(f"    - [모델명]_feature_importance.csv")
print(f"    - average_feature_importance.csv")
print(f"    - feature_importance_metadata.json")
print()
print(f"  🎯 예측 결과: {results_dir}/predictions/")
print(f"    - [모델명]_prediction_probabilities.csv")
print(f"    - [모델명]_complete_predictions.csv")
print(f"    - all_models_prediction_summary.csv")
print(f"    - prediction_probability_metadata.json")
print(f"    - prediction_results_metadata.json")

print(f"\n✅ 모든 데이터 추출이 완료되었습니다!")
print(f"📊 총 {len(basic_models)}개 모델의 결과가 저장되었습니다.")
print(f"🎯 다른 모델에서 활용할 수 있는 예측 결과가 준비되었습니다.")
print("="*60)

# --------------------------------------------------
# 10) 최적 모델: ADASYN + Stacking 조합
# --------------------------------------------------
print("\n" + "="*60)
print("=== 최적 모델: ADASYN + Stacking 조합 학습 ===")
print("="*60)

print("🎯 ADASYN 샘플링과 Stacking 모델을 조합한 최적 모델을 학습합니다...")
print("📊 이 조합이 가장 좋은 성능을 보일 것으로 예상됩니다.")

# ADASYN 샘플링 적용
print("1/3: ADASYN 샘플링 적용 중...")
X_tr_preprocessed = preprocessor.transform(X_tr)
X_resampled, y_resampled = ADASYN(random_state=42).fit_resample(X_tr_preprocessed, y_tr)

print(f"  - 원본 데이터 크기: {X_tr_preprocessed.shape}")
print(f"  - 샘플링 후 데이터 크기: {X_resampled.shape}")
print(f"  - 클래스별 샘플 수:")
for cls, count in pd.Series(y_resampled).value_counts().items():
    print(f"    {cls}: {count}개")

# 샘플링 후 가중치 계산
print("2/3: 샘플링 후 가중치 계산 중...")
le_resampled = LabelEncoder()
y_resampled_encoded = le_resampled.fit_transform(y_resampled)
class_counts_resampled = np.bincount(y_resampled_encoded)
total_samples_resampled = len(y_resampled_encoded)
resampled_weights = (total_samples_resampled / (len(class_counts_resampled) * class_counts_resampled)) ** 1.5
sample_weights_resampled = np.array([resampled_weights[label] for label in y_resampled_encoded])

print(f"  - 가중치 범위: {resampled_weights.min():.2f} ~ {resampled_weights.max():.2f}")

# Stacking 모델 구성 (기존 모델들 사용)
print("3/3: ADASYN + Stacking 모델 학습 중...")

# 개별 모델들 (가중치 적용 가능한 모델들)
base_estimators = [
    ('xgb', XGBWrapper(
        eval_metric="mlogloss",
        random_state=42,
        tree_method="hist",
        device="cuda",
        max_bin=256,
        single_precision_histogram=True,
        enable_categorical=False,
        max_leaves=0,
        grow_policy="lossguide",
        n_estimators=400,  # 최적 파라미터 사용
        max_depth=6,
        learning_rate=0.1,
        reg_alpha=0.1,
        reg_lambda=0.1
    )),
    ('lgb', LGBMClassifier(
        objective="multiclass",
        random_state=42,
        verbose=-1,
        device="gpu",
        gpu_platform_id=0,
        gpu_device_id=0,
        force_col_wise=True,
        gpu_use_dp=False,
        max_bin=255,
        num_leaves=31,
        min_child_samples=20,
        subsample=1.0,
        colsample_bytree=1.0,
        n_jobs=1,
        deterministic=True,
        force_row_wise=False,
        n_estimators=400,  # 최적 파라미터 사용
        max_depth=6,
        learning_rate=0.1
    )),
    ('rf', RandomForestClassifier(
        class_weight="balanced", 
        random_state=42,
        n_estimators=200,  # 최적 파라미터 사용
        max_depth=20,
        min_samples_split=5
    )),
    ('gb', GradientBoostingClassifier(
        random_state=42,
        n_estimators=200,  # 최적 파라미터 사용
        max_depth=6,
        learning_rate=0.1
    ))
]

# Stacking 모델 생성
adasyn_stacking = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(max_iter=5000, random_state=42),
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=1  # GPU 모델과 혼재로 인해 단일 스레드 사용
)

print(f"  - 베이스 모델: {[name for name, _ in base_estimators]}")
print(f"  - 메타 모델: LogisticRegression")
print(f"  - 교차검증: 5-fold StratifiedKFold")

# GPU 사용률 확인
try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        gpu_util = result.stdout.strip()
        print(f"🖥️ 학습 전 GPU 사용률: {gpu_util}%")
except:
    pass

# 모델 학습
print("  - 모델 학습 시작...")
adasyn_stacking.fit(X_resampled, y_resampled)

# 학습 후 GPU 사용률 확인
try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        gpu_util = result.stdout.strip()
        print(f"🖥️ 학습 후 GPU 사용률: {gpu_util}%")
except:
    pass

print("✅ ADASYN + Stacking 모델 학습 완료!")

# --------------------------------------------------
# 11) 최적 모델 성능 평가 및 데이터 추출
# --------------------------------------------------
print("\n" + "="*60)
print("=== 최적 모델 성능 평가 및 데이터 추출 ===")
print("="*60)

# 성능 평가
print("1/4: 최적 모델 성능 평가 중...")
X_te_preprocessed = preprocessor.transform(X_te)
y_pred_optimal = adasyn_stacking.predict(X_te_preprocessed)
proba_optimal = adasyn_stacking.predict_proba(X_te_preprocessed)

# 성능 지표 계산
acc_optimal = accuracy_score(y_te, y_pred_optimal, sample_weight=w_te)
macro_f1_optimal = f1_score(y_te, y_pred_optimal, average="macro", sample_weight=w_te)
weighted_f1_optimal = f1_score(y_te, y_pred_optimal, average="weighted", sample_weight=w_te)
bal_acc_optimal = balanced_accuracy_score(y_te, y_pred_optimal, sample_weight=w_te)
top3_acc_optimal = top_k_accuracy_score(y_te, proba_optimal, k=3, sample_weight=w_te)

# 클래스 이름 가져오기
class_names_optimal = getattr(adasyn_stacking, "classes_", None)
if class_names_optimal is None:
    class_names_optimal = np.unique(y_te)

y_bin_optimal = label_binarize(y_te, classes=class_names_optimal)
roc_auc_optimal = roc_auc_score(y_bin_optimal, proba_optimal, average="macro", sample_weight=w_te)

print(f"✅ 최적 모델 성능:")
print(f"  - 정확도: {acc_optimal:.4f}")
print(f"  - Macro F1: {macro_f1_optimal:.4f}")
print(f"  - Weighted F1: {weighted_f1_optimal:.4f}")
print(f"  - Balanced Accuracy: {bal_acc_optimal:.4f}")
print(f"  - Top-3 Accuracy: {top3_acc_optimal:.4f}")
print(f"  - Macro ROC-AUC: {roc_auc_optimal:.4f}")

# 2. 성능 비교에 최적 모델 추가
print("2/4: 성능 비교 데이터에 최적 모델 추가 중...")

optimal_performance = {
    "model_name": "ADASYN_Stacking",
    "accuracy": acc_optimal,
    "macro_f1": macro_f1_optimal,
    "weighted_f1": weighted_f1_optimal,
    "balanced_accuracy": bal_acc_optimal,
    "top3_accuracy": top3_acc_optimal,
    "top5_accuracy": top_k_accuracy_score(y_te, proba_optimal, k=5, sample_weight=w_te),
    "roc_auc_macro": roc_auc_optimal,
    "roc_auc_weighted": roc_auc_score(y_bin_optimal, proba_optimal, average="weighted", sample_weight=w_te),
    "f1_scores_by_class": f1_score(y_te, y_pred_optimal, average=None, sample_weight=w_te).tolist(),
    "classes": class_names_optimal.tolist()
}

# 기존 성능 결과에 최적 모델 추가
performance_results.append(optimal_performance)
performance_df_updated = pd.DataFrame(performance_results)
performance_df_updated.to_csv(f"{results_dir}/performance/model_performance_comparison_with_optimal.csv", index=False, encoding='utf-8-sig')

# 3. 예측 확률 분포에 최적 모델 추가
print("3/4: 예측 확률 분포에 최적 모델 추가 중...")

# 예측 확률 데이터프레임 생성
proba_df_optimal = pd.DataFrame(proba_optimal, columns=class_names_optimal)
proba_df_optimal['actual_class'] = y_te.values
proba_df_optimal['predicted_class'] = y_pred_optimal
proba_df_optimal['confidence'] = proba_optimal.max(axis=1)
proba_df_optimal['model'] = 'ADASYN_Stacking'

# Top-3 예측
top3_indices_optimal = np.argsort(proba_optimal, axis=1)[:, -3:][:, ::-1]
top3_classes_optimal = class_names_optimal[top3_indices_optimal]
top3_probs_optimal = np.take_along_axis(proba_optimal, top3_indices_optimal, axis=1)

proba_df_optimal['top1_class'] = top3_classes_optimal[:, 0]
proba_df_optimal['top1_prob'] = top3_probs_optimal[:, 0]
proba_df_optimal['top2_class'] = top3_classes_optimal[:, 1]
proba_df_optimal['top2_prob'] = top3_probs_optimal[:, 1]
proba_df_optimal['top3_class'] = top3_classes_optimal[:, 2]
proba_df_optimal['top3_prob'] = top3_probs_optimal[:, 2]

# 신뢰도 구간별 분석
confidence_bins = [0, 0.5, 0.7, 0.8, 0.9, 1.0]
confidence_labels = ['0-0.5', '0.5-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
proba_df_optimal['confidence_bin'] = pd.cut(proba_df_optimal['confidence'], bins=confidence_bins, labels=confidence_labels)

# 신뢰도 구간별 정확도
confidence_accuracy_optimal = proba_df_optimal.groupby('confidence_bin').apply(
    lambda x: (x['actual_class'] == x['predicted_class']).mean()
).reset_index()
confidence_accuracy_optimal.columns = ['confidence_bin', 'accuracy']

# 예측 확률 메타데이터에 최적 모델 추가
prediction_probability_results['ADASYN_Stacking'] = {
    "model_name": "ADASYN_Stacking",
    "class_names": class_names_optimal.tolist(),
    "confidence_accuracy": confidence_accuracy_optimal.to_dict('records'),
    "probability_stats": {
        "mean_confidence": proba_df_optimal['confidence'].mean(),
        "std_confidence": proba_df_optimal['confidence'].std(),
        "min_confidence": proba_df_optimal['confidence'].min(),
        "max_confidence": proba_df_optimal['confidence'].max()
    }
}

# 개별 CSV 파일로 저장
proba_df_optimal.to_csv(f"{results_dir}/predictions/ADASYN_Stacking_prediction_probabilities.csv", index=False, encoding='utf-8-sig')

# 4. 클래스별 성능 분석에 최적 모델 추가
print("4/4: 클래스별 성능 분석에 최적 모델 추가 중...")

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

precision_optimal, recall_optimal, f1_optimal, support_optimal = precision_recall_fscore_support(
    y_te, y_pred_optimal, labels=class_names_optimal, average=None, sample_weight=w_te
)

# 클래스별 정확도 계산
class_accuracy_optimal = {}
for i, class_name in enumerate(class_names_optimal):
    class_mask = (y_te == class_name)
    if class_mask.sum() > 0:
        class_accuracy_optimal[class_name] = (y_pred_optimal[class_mask] == y_te[class_mask]).mean()
    else:
        class_accuracy_optimal[class_name] = 0.0

# 클래스별 성능 데이터프레임
class_perf_df_optimal = pd.DataFrame({
    'class_name': class_names_optimal,
    'precision': precision_optimal,
    'recall': recall_optimal,
    'f1_score': f1_optimal,
    'support': support_optimal,
    'accuracy': [class_accuracy_optimal.get(cls, 0.0) for cls in class_names_optimal]
})

# 혼동 행렬 계산
cm_optimal = confusion_matrix(y_te, y_pred_optimal, labels=class_names_optimal, sample_weight=w_te)
cm_df_optimal = pd.DataFrame(cm_optimal, index=class_names_optimal, columns=class_names_optimal)

# 클래스별 성능 메타데이터에 최적 모델 추가
class_performance_results['ADASYN_Stacking'] = {
    "model_name": "ADASYN_Stacking",
    "class_performance": class_perf_df_optimal.to_dict('records'),
    "confusion_matrix": cm_df_optimal.to_dict(),
    "overall_metrics": {
        "macro_precision": precision_optimal.mean(),
        "macro_recall": recall_optimal.mean(),
        "macro_f1": f1_optimal.mean(),
        "weighted_precision": np.average(precision_optimal, weights=support_optimal),
        "weighted_recall": np.average(recall_optimal, weights=support_optimal),
        "weighted_f1": np.average(f1_optimal, weights=support_optimal)
    }
}

# 개별 CSV 파일로 저장
class_perf_df_optimal.to_csv(f"{results_dir}/performance/ADASYN_Stacking_class_performance.csv", index=False, encoding='utf-8-sig')
cm_df_optimal.to_csv(f"{results_dir}/performance/ADASYN_Stacking_confusion_matrix.csv", encoding='utf-8-sig')

# 5. 완전한 예측 결과에 최적 모델 추가
print("5/5: 완전한 예측 결과에 최적 모델 추가 중...")

# 결과 데이터프레임 생성
result_df_optimal = X_te.copy()
result_df_optimal['actual_class'] = y_te.values
result_df_optimal['predicted_class'] = y_pred_optimal
result_df_optimal['prediction_correct'] = (y_te == y_pred_optimal)
result_df_optimal['sample_weight'] = w_te.values
result_df_optimal['model'] = 'ADASYN_Stacking'

# 예측 확률 추가
for i, class_name in enumerate(class_names_optimal):
    result_df_optimal[f'prob_{class_name}'] = proba_optimal[:, i]
result_df_optimal['confidence'] = proba_optimal.max(axis=1)

# Top-3 예측
result_df_optimal['top1_class'] = top3_classes_optimal[:, 0]
result_df_optimal['top1_prob'] = top3_probs_optimal[:, 0]
result_df_optimal['top2_class'] = top3_classes_optimal[:, 1]
result_df_optimal['top2_prob'] = top3_probs_optimal[:, 1]
result_df_optimal['top3_class'] = top3_classes_optimal[:, 2]
result_df_optimal['top3_prob'] = top3_probs_optimal[:, 2]

# 원본 데이터의 추가 정보도 포함
test_indices = X_te.index
original_data = train.loc[test_indices]

for col in ['년도', '구분', '지역', '상병코드', '진료과', 'count']:
    if col in original_data.columns:
        result_df_optimal[f'original_{col}'] = original_data[col].values

# 예측 결과 메타데이터에 최적 모델 추가
prediction_results['ADASYN_Stacking'] = {
    "model_name": "ADASYN_Stacking",
    "total_predictions": len(result_df_optimal),
    "correct_predictions": result_df_optimal['prediction_correct'].sum(),
    "accuracy": result_df_optimal['prediction_correct'].mean(),
    "class_names": class_names_optimal.tolist() if isinstance(class_names_optimal, np.ndarray) else class_names_optimal
}

# 개별 CSV 파일로 저장
result_df_optimal.to_csv(f"{results_dir}/predictions/ADASYN_Stacking_complete_predictions.csv", index=False, encoding='utf-8-sig')

# 업데이트된 전체 예측 결과 요약
all_predictions_summary_updated = pd.DataFrame(prediction_results.values())
all_predictions_summary_updated.to_csv(f"{results_dir}/predictions/all_models_prediction_summary_with_optimal.csv", index=False, encoding='utf-8-sig')

# 업데이트된 메타데이터 저장
with open(f"{results_dir}/performance/performance_metadata_with_optimal.json", 'w', encoding='utf-8') as f:
    json.dump({
        "timestamp": datetime.now().isoformat(),
        "test_data_size": len(X_te),
        "train_data_size": len(X_tr),
        "num_classes": len(np.unique(y_te)),
        "class_distribution": y_te.value_counts().to_dict(),
        "models": performance_results,
        "optimal_model": {
            "name": "ADASYN_Stacking",
            "sampling_method": "ADASYN",
            "base_models": [name for name, _ in base_estimators],
            "meta_model": "LogisticRegression",
            "performance": optimal_performance
        }
    }, f, ensure_ascii=False, indent=2)

with open(f"{results_dir}/predictions/prediction_probability_metadata_with_optimal.json", 'w', encoding='utf-8') as f:
    json.dump(prediction_probability_results, f, ensure_ascii=False, indent=2)

with open(f"{results_dir}/performance/class_performance_metadata_with_optimal.json", 'w', encoding='utf-8') as f:
    json.dump(class_performance_results, f, ensure_ascii=False, indent=2)

with open(f"{results_dir}/predictions/prediction_results_metadata_with_optimal.json", 'w', encoding='utf-8') as f:
    json.dump(prediction_results, f, ensure_ascii=False, indent=2)

# --------------------------------------------------
# 최종 요약
# --------------------------------------------------
print("\n" + "="*60)
print("=== 최적 모델 추가 완료 ===")
print("="*60)

print(f"🎯 최적 모델: ADASYN + Stacking")
print(f"📊 성능 지표:")
print(f"  - 정확도: {acc_optimal:.4f}")
print(f"  - Macro F1: {macro_f1_optimal:.4f}")
print(f"  - Top-3 정확도: {top3_acc_optimal:.4f}")
print(f"  - ROC-AUC: {roc_auc_optimal:.4f}")

print(f"\n📁 추가된 파일들:")
print(f"  📊 성능 비교:")
print(f"    - model_performance_comparison_with_optimal.csv")
print(f"    - performance_metadata_with_optimal.json")
print(f"  🎯 예측 결과:")
print(f"    - ADASYN_Stacking_prediction_probabilities.csv")
print(f"    - ADASYN_Stacking_complete_predictions.csv")
print(f"    - ADASYN_Stacking_class_performance.csv")
print(f"    - ADASYN_Stacking_confusion_matrix.csv")
print(f"    - all_models_prediction_summary_with_optimal.csv")

print(f"\n✅ 최적 모델이 모든 데이터 추출에 포함되었습니다!")
print(f"🎯 ADASYN_Stacking_complete_predictions.csv 파일에서 최적 모델의 예측 결과를 확인할 수 있습니다.")
print("="*60)