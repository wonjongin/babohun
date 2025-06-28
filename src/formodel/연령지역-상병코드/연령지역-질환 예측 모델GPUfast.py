import pandas as pd
import numpy as np
import warnings
import os
import json
import subprocess
import multiprocessing
from datetime import datetime

# CUDA 환경 변수 설정 (GPU 사용 강제)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XGBOOST_USE_CUDA'] = '1'
os.environ['LIGHTGBM_USE_GPU'] = '1'

# 경고 무시
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, top_k_accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize

from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTEENN

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

print("🚀 향상된 테스트용 모델 학습 시작!")
print("="*60)

# --------------------------------------------------
# GPU 및 시스템 정보 확인
# --------------------------------------------------
print("🔧 시스템 정보 확인 중...")

# GPU 사용 가능 여부 확인
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ GPU 사용 가능: NVIDIA GPU가 감지되었습니다.")
        
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
            
    else:
        print("⚠️ GPU 사용 불가: NVIDIA GPU가 감지되지 않았습니다.")
        print("CPU 모드로 실행됩니다.")
except:
    print("⚠️ GPU 사용 불가: nvidia-smi 명령어를 찾을 수 없습니다.")
    print("CPU 모드로 실행됩니다.")

# CPU 코어 수 확인
cpu_count = multiprocessing.cpu_count()
print(f"🔧 CPU 코어 수: {cpu_count}개")
print(f"🔧 멀티코어 활용: 활성화")

# 환경 변수 확인
print(f"🔧 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print(f"🔧 XGBOOST_USE_CUDA: {os.environ.get('XGBOOST_USE_CUDA', 'Not set')}")
print(f"🔧 LIGHTGBM_USE_GPU: {os.environ.get('LIGHTGBM_USE_GPU', 'Not set')}")

print()

# --------------------------------------------------
# 1) 데이터 적재 및 가공 (샘플링)
# --------------------------------------------------
print("1/8: 데이터 로드 및 샘플링 중...")

# 전체 데이터 로드
df = pd.read_csv("new_merged_data/df_result2_with_심평원.csv", dtype=str)
print(f"  - 전체 데이터 크기: {df.shape}")

# 데이터 샘플링 (50%로 증가)
sample_size = int(len(df) * 0.5)
df_sample = df.sample(n=sample_size, random_state=42)
print(f"  - 샘플링 후 데이터 크기: {df_sample.shape}")

age_cols = ["59이하", "60-64", "65-69", "70-79", "80-89", "90이상"]

m = df_sample.melt(
    id_vars=["년도", "구분", "지역", "상병코드", "진료과"],
    value_vars=age_cols,
    var_name="age_group",
    value_name="count",
)
m["count"] = pd.to_numeric(m["count"], errors="coerce").fillna(0).astype(int)
m["대표진료과"] = m["진료과"]
train = m[m["대표진료과"].notna()]

print(f"  - 최종 학습 데이터 크기: {train.shape}")
print(f"  - 클래스 수: {len(train['대표진료과'].unique())}")
print(f"  - 클래스별 샘플 수:")
for cls, count in train['대표진료과'].value_counts().head(10).items():
    print(f"    {cls}: {count}개")

# 강화된 피처 엔지니어링 (원래 코드와 동일)
train["year_num"] = train["년도"].astype(int) - train["년도"].astype(int).min()

age_mapping = {
    "59이하": 30, "60-64": 62, "65-69": 67, 
    "70-79": 75, "80-89": 85, "90이상": 95
}
train["age_num"] = train["age_group"].map(age_mapping)

major_cities = ["서울", "부산", "대구", "인천", "광주", "대전"]
train["is_major_city"] = train["지역"].isin(major_cities).astype(int)
train["is_inpatient"] = (train["구분"] == "입원").astype(int)
train["disease_group"] = train["상병코드"].str[:3]

# 추가 피처 (원래 코드와 동일)
train["year_trend"] = train["year_num"] ** 2
train["age_city_interaction"] = train["age_num"] * train["is_major_city"]
train["age_year_interaction"] = train["age_num"] * train["year_num"]
train["region_age"] = train["지역"] + "_" + train["age_group"]

X = train[["year_num", "age_num", "is_major_city", "is_inpatient", 
           "year_trend", "age_city_interaction", "age_year_interaction",
           "지역", "age_group", "구분", "disease_group", "region_age"]]
y = train["대표진료과"]
w = train["count"]

# --------------------------------------------------
# 2) 학습 / 검증 분리
# --------------------------------------------------
print("2/8: 데이터 분할 중...")
X_tr, X_te, y_tr, y_te, w_tr, w_te = train_test_split(
    X, y, w, test_size=0.20, stratify=y, random_state=42
)

print(f"  - 학습 데이터: {X_tr.shape}")
print(f"  - 테스트 데이터: {X_te.shape}")

# --------------------------------------------------
# 3) 전처리 파이프라인
# --------------------------------------------------
print("3/8: 전처리 파이프라인 설정 중...")

num_cols = ["year_num", "age_num", "is_major_city", "is_inpatient", 
            "year_trend", "age_city_interaction", "age_year_interaction"]
cat_cols = ["지역", "age_group", "구분", "disease_group", "region_age"]

preprocessor = ColumnTransformer(
    [
        ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_cols),
        ("scale", StandardScaler(), num_cols),
    ]
)

# --------------------------------------------------
# 4) XGBWrapper 클래스 정의
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
print("4/8: 파이프라인 및 그리드 정의 중...")

def make_pipeline(clf, param_grid):
    pipe = ImbPipeline(
        [
            ("prep", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("variance", VarianceThreshold(threshold=0.01)),
            ("select", SelectKBest(f_classif)),
            ("clf", clf),
        ]
    )
    return pipe, param_grid

# k 값을 동적으로 설정
preprocessor.fit(X_tr)
n_features_after_prep = len(preprocessor.get_feature_names_out())
max_k = min(n_features_after_prep, 50)  # 테스트용으로 50개로 제한

# 멀티코어 활용 설정
n_jobs = max(1, int(cpu_count * 0.75))  # 75% 코어 활용
print(f"  - 멀티코어 활용: {n_jobs}개 코어 사용")

# 모델별 파이프라인 및 그리드 정의 (GPU 최적화 포함)
pipe_lr, params_lr = make_pipeline(
    LogisticRegression(penalty="l1", solver="saga", max_iter=1000, class_weight="balanced"),
    {
        "select__k": [max_k//4, max_k//2],
        "clf__C": [0.1, 1, 10],
    },
)

pipe_rf, params_rf = make_pipeline(
    RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=n_jobs),
    {
        "select__k": [max_k//4, max_k//2],
        "clf__n_estimators": [50, 100],
        "clf__max_depth": [10, 20],
    },
)

pipe_xgb, params_xgb = make_pipeline(
    XGBWrapper(
        eval_metric="mlogloss",
        random_state=42,
        tree_method="hist",  # GPU 최적화
        device="cuda",  # GPU 사용
        max_bin=256,  # GPU 메모리 최적화
        single_precision_histogram=True,  # GPU 메모리 절약
        enable_categorical=False,  # 카테고리형 비활성화
        max_leaves=0,  # GPU 최적화
        grow_policy="lossguide",  # GPU 최적화
    ),
    {
        "select__k": [max_k//4, max_k//2],
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [3, 6],
        "clf__learning_rate": [0.1, 0.2],
    },
)

pipe_lgb, params_lgb = make_pipeline(
    LGBMClassifier(
        objective="multiclass",
        random_state=42,
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
        n_jobs=1,  # GPU 사용시 단일 스레드
        deterministic=True,  # 재현성 보장
        force_row_wise=False,  # GPU 최적화
    ),
    {
        "select__k": [max_k//4, max_k//2],
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [3, 6],
        "clf__learning_rate": [0.1, 0.2],
    },
)

pipe_gb, params_gb = make_pipeline(
    GradientBoostingClassifier(random_state=42),
    {
        "select__k": [max_k//4, max_k//2],
        "clf__n_estimators": [50, 100],
        "clf__max_depth": [3, 6],
        "clf__learning_rate": [0.1, 0.2],
    },
)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # 테스트용으로 3-fold

# --------------------------------------------------
# 6) 그리드 서치 실행
# --------------------------------------------------
print("5/8: 그리드 서치 실행 중...")

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
    print(f"  - 모델 {i}/5: {name.upper()} 그리드 서치 중...")
    
    # GPU 사용 확인 (XGBoost, LightGBM의 경우)
    if name in ['xgb', 'lgb']:
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                gpu_util = result.stdout.strip()
                print(f"    🖥️ 학습 전 GPU 사용률: {gpu_util}%")
        except:
            pass
    
    # 멀티코어 활용 (CPU 모델들)
    if name in ['lr', 'rf', 'gb']:
        grid = GridSearchCV(pipe, params, cv=cv, scoring="accuracy", n_jobs=n_jobs, verbose=0)
    else:
        # GPU 모델들은 단일 스레드 (GPU 사용)
        grid = GridSearchCV(pipe, params, cv=cv, scoring="accuracy", n_jobs=1, verbose=0)
    
    grid.fit(X_tr, y_tr)
    grids[name] = grid
    
    # 학습 후 GPU 사용률 확인
    if name in ['xgb', 'lgb']:
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                gpu_util = result.stdout.strip()
                print(f"    🖥️ 학습 후 GPU 사용률: {gpu_util}%")
        except:
            pass
    
    print(f"    ✅ {name.upper()} 최적 점수: {grid.best_score_:.4f}")

# --------------------------------------------------
# 7) 앙상블 (Voting & Stacking)
# --------------------------------------------------
print("6/8: 앙상블 모델 학습 중...")

# Voting Classifier
print("  - Voting Classifier 학습 중...")
estimators = [(n, grids[n].best_estimator_) for n in ["lr", "rf", "xgb", "lgb", "gb"]]
voting = VotingClassifier(estimators=estimators, voting="soft")
voting.fit(X_tr, y_tr)

# Stacking Classifier
print("  - Stacking Classifier 학습 중...")
stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=cv,
    n_jobs=n_jobs,  # 멀티코어 활용
)
stack.fit(X_tr, y_tr)

# --------------------------------------------------
# 8) 평가 함수
# --------------------------------------------------
print("7/8: 모델 평가 중...")

def eval_model(name, model, X, y_true, w):
    y_pred = model.predict(X)
    proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_true, y_pred, sample_weight=w)
    macro_f1 = f1_score(y_true, y_pred, average="macro", sample_weight=w)
    
    result = {
        "model": name,
        "accuracy": acc,
        "macro_f1": macro_f1
    }
    
    if proba is not None:
        class_order = getattr(model, "orig_classes_", None)
        if class_order is None:
            class_order = getattr(model, "classes_", None)
        if class_order is None:
            class_order = np.unique(y_true)
        
        top3_acc = top_k_accuracy_score(y_true, proba, k=3, sample_weight=w)
        bal_acc = balanced_accuracy_score(y_true, y_pred, sample_weight=w)
        
        y_bin = label_binarize(y_true, classes=class_order)
        roc_auc = roc_auc_score(y_bin, proba, average="macro", sample_weight=w)
        
        result.update({
            "top3_accuracy": top3_acc,
            "balanced_accuracy": bal_acc,
            "roc_auc": roc_auc
        })
    
    return result

# 모든 모델 평가
results = []

# 기본 모델들
for name, grid in grids.items():
    result = eval_model(name.upper(), grid.best_estimator_, X_te, y_te, w_te)
    results.append(result)
    print(f"  ✅ {name.upper()}: 정확도={result['accuracy']:.4f}, F1={result['macro_f1']:.4f}")

# 앙상블 모델들
voting_result = eval_model("Voting", voting, X_te, y_te, w_te)
results.append(voting_result)
print(f"  ✅ Voting: 정확도={voting_result['accuracy']:.4f}, F1={voting_result['macro_f1']:.4f}")

stacking_result = eval_model("Stacking", stack, X_te, y_te, w_te)
results.append(stacking_result)
print(f"  ✅ Stacking: 정확도={stacking_result['accuracy']:.4f}, F1={stacking_result['macro_f1']:.4f}")

# --------------------------------------------------
# 9) 샘플링 기법 테스트
# --------------------------------------------------
print("8/8: 샘플링 기법 테스트 중...")

sampling_methods = {
    'adasyn': ADASYN(random_state=42, n_neighbors=3),
    'borderline_smote': BorderlineSMOTE(random_state=42, k_neighbors=3),
    'smote_enn': SMOTEENN(random_state=42)
}

# 전처리 적용
X_tr_preprocessed = preprocessor.fit_transform(X_tr)
X_te_preprocessed = preprocessor.transform(X_te)

for name, sampler in sampling_methods.items():
    try:
        print(f"  - {name.upper()} 테스트 중...")
        
        # 샘플링 적용
        X_resampled, y_resampled = sampler.fit_resample(X_tr_preprocessed, y_tr)
        
        # GPU 최적화된 XGBoost 모델로 성능 측정
        simple_model = XGBWrapper(
            n_estimators=100, 
            random_state=42,
            tree_method="hist",
            device="cuda",
            max_bin=256,
            single_precision_histogram=True,
            enable_categorical=False,
            max_leaves=0,
            grow_policy="lossguide",
        )
        simple_model.fit(X_resampled, y_resampled)
        
        result = eval_model(f"{name.upper()}_sampling", simple_model, X_te_preprocessed, y_te, w_te)
        results.append(result)
        print(f"    ✅ {name.upper()}: 정확도={result['accuracy']:.4f}, F1={result['macro_f1']:.4f}")
        
    except Exception as e:
        print(f"    ❌ {name.upper()}: 샘플링 실패 - {str(e)}")

# --------------------------------------------------
# 10) 결과 정리 및 저장
# --------------------------------------------------
print("\n📊 결과 정리 및 저장 중...")

# 결과 데이터프레임 생성
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("macro_f1", ascending=False)

print(f"\n📊 최종 결과 (F1-score 순):")
print(results_df.to_string(index=False))

# 최고 성능 모델 찾기
best_model = results_df.iloc[0]
print(f"\n🏆 최고 성능 모델: {best_model['model']}")
print(f"   - 정확도: {best_model['accuracy']:.4f}")
print(f"   - F1-score: {best_model['macro_f1']:.4f}")
if 'top3_accuracy' in best_model:
    print(f"   - Top-3 정확도: {best_model['top3_accuracy']:.4f}")
if 'roc_auc' in best_model:
    print(f"   - ROC-AUC: {best_model['roc_auc']:.4f}")

# 결과 저장
test_results_dir = "test_results"
os.makedirs(test_results_dir, exist_ok=True)

# CSV 저장
results_df.to_csv(f"{test_results_dir}/enhanced_test_results.csv", index=False, encoding='utf-8-sig')

# JSON 메타데이터 저장
metadata = {
    "timestamp": datetime.now().isoformat(),
    "test_type": "enhanced_test",
    "system_info": {
        "cpu_cores": cpu_count,
        "gpu_available": result.returncode == 0 if 'result' in locals() else False,
        "multicore_utilization": f"{n_jobs}/{cpu_count} cores"
    },
    "data_info": {
        "original_size": len(df),
        "sampled_size": len(df_sample),
        "final_train_size": len(train),
        "test_size": len(X_te),
        "num_classes": len(train['대표진료과'].unique()),
        "features": list(X.columns),
        "features_after_preprocessing": n_features_after_prep
    },
    "best_model": {
        "name": best_model['model'],
        "accuracy": best_model['accuracy'],
        "f1_score": best_model['macro_f1']
    },
    "all_results": results,
    "grid_search_info": {
        "cv_folds": 3,
        "models_tested": list(grids.keys()),
        "max_features_selected": max_k,
        "multicore_jobs": n_jobs
    }
}

with open(f"{test_results_dir}/enhanced_test_metadata.json", 'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print(f"\n✅ 향상된 테스트 완료!")
print(f"📁 결과 저장 위치: {test_results_dir}/")
print(f"📊 총 {len(results)}개 모델 테스트 완료")
print(f"⏱️ 예상 전체 학습 시간: 약 10-30분 (현재 테스트: 약 3-5분)")

# 방향성 제안
print(f"\n💡 방향성 제안:")
if best_model['model'].endswith('_sampling'):
    print(f"  - 샘플링 기법이 효과적입니다.")
elif best_model['model'] in ['Voting', 'Stacking']:
    print(f"  - 앙상블이 효과적입니다.")
elif best_model['model'] in ['XGB', 'LGB']:
    print(f"  - 부스팅 모델이 좋은 성능을 보입니다.")

print(f"  - 전체 데이터로 학습하면 성능이 더 향상될 것으로 예상됩니다.")

print("="*60)
print("🎯 이제 전체 데이터로 본격적인 학습을 진행하세요!")
print("="*60) 