import pandas as pd
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
import subprocess
import os
import multiprocessing

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
major_cities = ["서울", "부산", "대구", "인천", "광주", "대전", "울산"]
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
        gpu_use_dp=False,  # 단정밀도 사용
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