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

# CUDA 환경 변수 설정 (GPU 사용 활성화 - 메모리 효율적)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XGBOOST_USE_CUDA'] = '1'
os.environ['LIGHTGBM_USE_GPU'] = '1'

# GPU 메모리 효율적 설정
print("🚀 GPU 모드로 실행 (메모리 효율적 설정)")
print("⚠️ 메모리 부족 시 자동으로 CPU 모드로 전환됩니다.")

# CPU 모드로 전환 (빠른 실행을 위해)
print("🚀 CPU 모드로 실행 (빠른 학습을 위해)")
print("⚠️ GPU 설정을 비활성화하여 안정적인 실행을 보장합니다.")

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

# 모든 경고 무시
warnings.filterwarnings("ignore")

# 특정 경고만 무시 (더 구체적으로)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.preprocessing._encoders")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.feature_selection._univariate_selection")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.feature_selection._univariate_selection")
warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn.linear_model._sag")

# XGBoost 관련 경고 무시
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", category=FutureWarning, module="xgboost")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="xgboost")

# LightGBM 관련 경고 무시
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
warnings.filterwarnings("ignore", category=FutureWarning, module="lightgbm")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="lightgbm")

# 추가적인 경고 무시
warnings.filterwarnings("ignore", message=".*FutureWarning.*")
warnings.filterwarnings("ignore", message=".*DeprecationWarning.*")
warnings.filterwarnings("ignore", message=".*UserWarning.*")

# sklearn 관련 모든 경고 무시
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="sklearn")
warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")

# pandas 관련 경고 무시
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pandas")

# numpy 관련 경고 무시
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")
warnings.filterwarnings("ignore", category=FutureWarning, module="numpy")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

# matplotlib 관련 경고 무시
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=FutureWarning, module="matplotlib")

# seaborn 관련 경고 무시
warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

# imblearn 관련 경고 무시
warnings.filterwarnings("ignore", category=UserWarning, module="imblearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="imblearn")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="imblearn")

# scipy 관련 경고 무시
warnings.filterwarnings("ignore", category=UserWarning, module="scipy")
warnings.filterwarnings("ignore", category=FutureWarning, module="scipy")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")

# 모든 모듈의 특정 경고 타입 무시
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# 특정 메시지 패턴 무시
warnings.filterwarnings("ignore", message=".*convergence.*")
warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")
warnings.filterwarnings("ignore", message=".*UserWarning.*")
warnings.filterwarnings("ignore", message=".*FutureWarning.*")
warnings.filterwarnings("ignore", message=".*DeprecationWarning.*")
warnings.filterwarnings("ignore", message=".*RuntimeWarning.*")
warnings.filterwarnings("ignore", message=".*The default value of.*")
warnings.filterwarnings("ignore", message=".*will change.*")
warnings.filterwarnings("ignore", message=".*is deprecated.*")
warnings.filterwarnings("ignore", message=".*will be removed.*")
warnings.filterwarnings("ignore", message=".*SettingWithCopyWarning.*")
warnings.filterwarnings("ignore", message=".*DataFrame.*")
warnings.filterwarnings("ignore", message=".*Series.*")
warnings.filterwarnings("ignore", message=".*numpy.*")
warnings.filterwarnings("ignore", message=".*pandas.*")
warnings.filterwarnings("ignore", message=".*sklearn.*")
warnings.filterwarnings("ignore", message=".*xgboost.*")
warnings.filterwarnings("ignore", message=".*lightgbm.*")
warnings.filterwarnings("ignore", message=".*imblearn.*")

# 특정 sklearn 경고들 무시
warnings.filterwarnings("ignore", message="k=.*is greater than n_features=.*")
warnings.filterwarnings("ignore", message=".*All the features will be returned.*")
warnings.filterwarnings("ignore", message="Found unknown categories in columns.*")
warnings.filterwarnings("ignore", message=".*These unknown categories will be encoded as all zeros.*")

# sklearn feature_selection 모듈의 특정 경고 무시
warnings.filterwarnings("ignore", message=".*k=.*is greater than n_features=.*", module="sklearn.feature_selection._univariate_selection")
warnings.filterwarnings("ignore", message=".*All the features will be returned.*", module="sklearn.feature_selection._univariate_selection")

# sklearn preprocessing 모듈의 특정 경고 무시
warnings.filterwarnings("ignore", message=".*Found unknown categories in columns.*", module="sklearn.preprocessing._encoders")
warnings.filterwarnings("ignore", message=".*These unknown categories will be encoded as all zeros.*", module="sklearn.preprocessing._encoders")

# 모든 sklearn 경고 무시 (최후의 수단)
warnings.filterwarnings("ignore", module="sklearn")

# NumPy 타입을 JSON 직렬화 가능한 타입으로 변환하는 함수
def convert_numpy_types(obj):
    """NumPy 타입을 Python 기본 타입으로 변환"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

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
df = pd.read_csv("new_merged_data/df_result2_with_심평원.csv", dtype=str)
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
max_k = min(n_features_after_prep, 50)  # 최대 50개로 축소 (100 → 50)

# Logistic Regression
pipe_lr, params_lr = make_pipeline(
    LogisticRegression(
        penalty="l1", solver="saga", max_iter=1000, class_weight="balanced"  # max_iter 대폭 축소
    ),
    {
        "select__k": [max_k],  # 단일 값으로 축소
        "clf__C": [0.1, 1],  # 조합 축소
    },
)

# Random Forest
pipe_rf, params_rf = make_pipeline(
    RandomForestClassifier(class_weight="balanced", random_state=42),
    {
        "select__k": [max_k],  # 단일 값으로 축소
        "clf__n_estimators": [100],  # 단일 값으로 축소
        "clf__max_depth": [None, 20],  # 조합 축소
        "clf__min_samples_split": [2],  # 단일 값으로 축소
    },
)

# XGBoost
pipe_xgb, params_xgb = make_pipeline(
    XGBWrapper(
        eval_metric="mlogloss",
        random_state=42,
        tree_method="hist",  # CPU 안전 모드
        n_jobs=-1,  # 모든 CPU 코어 사용
        max_bin=128,  # 메모리 절약
        enable_categorical=False,  # 카테고리형 비활성화
        max_leaves=0,  # 기본값 사용
        grow_policy="lossguide",  # 기본 정책
    ),
    {
        "select__k": [max_k],  # 단일 값으로 축소
        "clf__n_estimators": [100],  # 단일 값으로 축소
        "clf__max_depth": [3, 6],  # 조합 축소
        "clf__learning_rate": [0.1],  # 단일 값으로 축소
        "clf__reg_alpha": [0],  # 단일 값으로 축소
        "clf__reg_lambda": [0],  # 단일 값으로 축소
    },
)

# LightGBM
pipe_lgb, params_lgb = make_pipeline(
    LGBMClassifier(
        objective="multiclass",
        random_state=42,
        class_weight="balanced",
        verbose=-1,
        n_jobs=-1,  # 모든 CPU 코어 사용
        force_col_wise=True,  # CPU 최적화
        max_bin=128,  # 메모리 절약
        num_leaves=31,  # 고정값으로 경고 제거
        min_child_samples=20,  # 고정값으로 경고 제거
        subsample=1.0,  # 고정값으로 경고 제거
        colsample_bytree=1.0,  # 고정값으로 경고 제거
        deterministic=True,  # 재현성 보장
    ),
    {
        "select__k": [max_k],  # 단일 값으로 축소
        "clf__n_estimators": [100],  # 단일 값으로 축소
        "clf__max_depth": [3, 6],  # 조합 축소
        "clf__learning_rate": [0.1],  # 단일 값으로 축소
        "clf__num_leaves": [31],  # 단일 값으로 축소
    },
)

# Gradient Boosting
pipe_gb, params_gb = make_pipeline(
    GradientBoostingClassifier(random_state=42),
    {
        "select__k": [max_k],  # 단일 값으로 축소
        "clf__n_estimators": [100],  # 단일 값으로 축소
        "clf__max_depth": [3, 6],  # 조합 축소
        "clf__learning_rate": [0.1],  # 단일 값으로 축소
    },
)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # 5-fold → 3-fold로 축소

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
    print(f"예상 학습 시간: 약 10-30초 (CPU 최적화)")  # CPU 모델
    
    # CPU 모델들: 멀티코어 활용
    n_jobs = max(1, int(cpu_count * 0.8))  # 80% 코어 활용
    
    grid = GridSearchCV(
        pipe, params, cv=cv, scoring="accuracy", n_jobs=n_jobs, verbose=0
    )
    print(f"GridSearchCV 시작...")
    grid.fit(X_tr, y_tr)  # sample_weight 미사용
    grids[name] = grid
    
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
n_jobs = max(1, int(cpu_count * 0.6))   # 60% 코어 활용 (50% → 60%)
stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=3000),  # max_iter 축소
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),  # 5-fold → 3-fold
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
        
        # 모델 학습
        print("모델 학습 중...")
        print("예상 학습 시간: 약 10-20초")
        
        # 간단한 모델로 테스트 (전처리된 데이터 사용)
        from sklearn.ensemble import RandomForestClassifier
        test_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        test_model.fit(X_resampled, y_resampled)
        
        # 평가
        print("✅ ADASYN 학습 완료")
        print("ADASYN 평가 중...")
        
        # 예측
        y_pred = test_model.predict(X_te_preprocessed)
        
        # 성능 평가
        acc = accuracy_score(y_te, y_pred, sample_weight=w_te)
        macro_f1 = f1_score(y_te, y_pred, average="macro", sample_weight=w_te)
        
        print(f"=== ADASYN 모델 평가 중... ===")
        print(f"\n=== ADASYN ===")
        print(f"Accuracy: {acc:.4f}")
        print(f"Macro-F1: {macro_f1:.4f}")
        
        if hasattr(test_model, 'predict_proba'):
            proba = test_model.predict_proba(X_te_preprocessed)
            top3_acc = top_k_accuracy_score(y_te, proba, k=3, sample_weight=w_te)
            bal_acc = balanced_accuracy_score(y_te, y_pred, sample_weight=w_te)
            print(f"Top-3 Accuracy: {top3_acc:.4f}")
            print(f"Balanced Accuracy: {bal_acc:.4f}")
            
            # ROC-AUC 계산 시도
            try:
                class_order = np.unique(y_te)
                y_bin = label_binarize(y_te, classes=class_order)
                roc_auc = roc_auc_score(y_bin, proba, average="macro", sample_weight=w_te)
                print(f"Macro ROC-AUC: {roc_auc:.4f}")
            except Exception as e:
                print(f"Macro ROC-AUC: nan")
        
        print("\nClassification Report:")
        print(classification_report(y_te, y_pred, sample_weight=w_te, digits=3))
        
    except Exception as e:
        print(f"❌ {name.upper()} 처리 중 오류 발생: {str(e)}")
        print("다음 샘플링 기법으로 진행...")
        continue

print("3/3: 모든 모델 학습 및 평가 완료")
print("="*60)
print("🎉 모든 분석이 완료되었습니다!")
print("="*60)

# --------------------------------------------------
# 데이터 추출 및 저장
# --------------------------------------------------
print("="*60)
print("=== 데이터 추출 및 저장 시작 ===")
print("="*60)

# 결과 저장 디렉토리
results_dir = "model_results"
os.makedirs(f"{results_dir}/performance", exist_ok=True)
os.makedirs(f"{results_dir}/features", exist_ok=True)
os.makedirs(f"{results_dir}/predictions", exist_ok=True)
os.makedirs(f"{results_dir}/models", exist_ok=True)

print(f"📁 결과 저장 디렉토리: {results_dir}/")

# 기본 모델들
basic_models = {
    "XGB": grids["xgb"].best_estimator_,
    "LGB": grids["lgb"].best_estimator_,
    "GB": grids["gb"].best_estimator_,
    "RF": grids["rf"].best_estimator_,
    "LR": grids["lr"].best_estimator_,
    "Voting": vot,
    "Stacking": stack
}

# 1) 모델 성능 비교 결과 저장
print("1/6: 모델 성능 비교 결과 저장 중...")

performance_results = {}

for name, model in basic_models.items():
    print(f"  - {name} 모델 성능 추출 중...")
    
    try:
        # 모델 성능 추출
        y_pred = model.predict(X_te)
        proba = model.predict_proba(X_te) if hasattr(model, "predict_proba") else None
        
        # 기본 지표
        acc = accuracy_score(y_te, y_pred, sample_weight=w_te)
        macro_f1 = f1_score(y_te, y_pred, average="macro", sample_weight=w_te)
        
        # 추가 지표
        bal_acc = balanced_accuracy_score(y_te, y_pred, sample_weight=w_te)
        
        if proba is not None:
            top3_acc = top_k_accuracy_score(y_te, proba, k=3, sample_weight=w_te)
            
            # ROC-AUC 계산
            try:
                class_order = getattr(model, "orig_classes_", None)
                if class_order is None:
                    class_order = getattr(model, "classes_", None)
                if class_order is None:
                    class_order = np.unique(y_te)
                
                y_bin = label_binarize(y_te, classes=class_order)
                roc_auc = roc_auc_score(y_bin, proba, average="macro", sample_weight=w_te)
            except:
                roc_auc = float('nan')
        else:
            top3_acc = float('nan')
            roc_auc = float('nan')
        
        performance_results[name] = {
            "model_name": name,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "top3_accuracy": top3_acc,
            "balanced_accuracy": bal_acc,
            "macro_roc_auc": roc_auc
        }
        
    except Exception as e:
        print(f"  - {name} 모델 성능 추출 실패: {str(e)}")

# 성능 비교 CSV 저장
if performance_results:
    performance_df = pd.DataFrame(performance_results).T
    performance_df.to_csv(f"{results_dir}/performance/model_performance_comparison.csv", encoding='utf-8-sig')

print(f"✅ 모델 성능 비교 결과 저장 완료: {results_dir}/performance/")

# 2) 그리드 서치 결과 저장
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
            "mean_test_score": convert_numpy_types(cv_results['mean_test_score'][i]),
            "std_test_score": convert_numpy_types(cv_results['std_test_score'][i]),
            "rank_test_score": convert_numpy_types(cv_results['rank_test_score'][i])
        })
        grid_summary["param_combinations"].append(param_combo)
    
    grid_search_results[name] = convert_numpy_types(grid_summary)

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

# 3) 피처 중요도 저장
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

# 4) 예측 확률 분포 저장
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

# 5) 클래스별 성능 분석 저장
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
            'accuracy': [class_accuracy[cls] for cls in class_names]
        })
        
        class_performance_results[name] = {
            "model_name": name,
            "class_performance": class_perf_df.to_dict('records'),
            "overall_stats": {
                "macro_precision": precision.mean(),
                "macro_recall": recall.mean(),
                "macro_f1": f1.mean(),
                "weighted_f1": np.average(f1, weights=support)
            }
        }
        
        # 개별 CSV 파일로 저장
        class_perf_df.to_csv(f"{results_dir}/performance/{name}_class_performance.csv", index=False, encoding='utf-8-sig')
        
    except Exception as e:
        print(f"  - {name} 클래스별 성능 분석 실패: {str(e)}")

# 클래스별 성능 메타데이터 저장
with open(f"{results_dir}/performance/class_performance_metadata.json", 'w', encoding='utf-8') as f:
    json.dump(class_performance_results, f, ensure_ascii=False, indent=2)

print(f"✅ 클래스별 성능 분석 저장 완료: {results_dir}/performance/")

# 6) 모델 저장
print("6/6: 모델 저장 중...")

import joblib

for name, model in basic_models.items():
    try:
        model_path = f"{results_dir}/models/{name}_model.pkl"
        joblib.dump(model, model_path)
        print(f"  - {name} 모델 저장 완료: {model_path}")
    except Exception as e:
        print(f"  - {name} 모델 저장 실패: {str(e)}")

print(f"✅ 모델 저장 완료: {results_dir}/models/")

print("\n" + "="*60)
print("🎉 모든 분석 및 저장이 완료되었습니다!")
print("="*60)
print(f"📁 결과 파일 위치: {results_dir}/")
print("📊 성능 비교: performance/model_performance_comparison.csv")
print("🔍 그리드 서치: performance/grid_search_results.json")
print("📈 피처 중요도: features/")
print("🎯 예측 확률: predictions/")
print("🏥 클래스별 성능: performance/class_performance_*.csv")
print("💾 모델 파일: models/")
print("="*60)
