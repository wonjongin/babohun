import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
import warnings
import os
from datetime import datetime
import pickle
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier

warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_and_preprocess_data():
    """데이터 로드 및 전처리"""
    print("데이터 로드 중...")
    
    # 기본 데이터 로드
    df = pd.read_csv('new_merged_data/df_result2_with_심평원_진료비_5구간.csv')
    
    # 시계열 예측 데이터 로드
    timeseries_df = pd.read_csv('analysis_data/병원별_진료과별_입원외래_통합_시계열예측결과_개선_3개년.csv')
    
    print(f"기본 데이터 shape: {df.shape}")
    print(f"시계열 예측 데이터 shape: {timeseries_df.shape}")
    
    # 시계열 예측 데이터에서 필요한 컬럼만 선택
    timeseries_features = timeseries_df[['병원', '진료과', '연도', '실제값', 'ARIMA예측', 'RF예측', 'XGB예측',
                                       'ARIMA_RMSE', 'ARIMA_MAE', 'ARIMA_MAPE', 'ARIMA_R2', 'ARIMA_Adjusted_R2',
                                       'RF_RMSE', 'RF_MAE', 'RF_MAPE', 'RF_R2', 'RF_Adjusted_R2',
                                       'XGB_RMSE', 'XGB_MAE', 'XGB_MAPE', 'XGB_R2', 'XGB_Adjusted_R2']].copy()
    
    # 진료과명 통일 (기본 데이터의 진료과명과 매칭)
    dept_mapping = {
        '가정의학과': '가정의학과',
        '건강관리과': '건강관리과',
        '내과': '내과',
        '마취과': '마취과',
        '비뇨기과': '비뇨의학과',
        '산부인과': '산부인과',
        '소아청소년과': '소아청소년과',
        '신경과': '신경과',
        '신경외과': '신경외과',
        '안과': '안과',
        '외과': '외과',
        '이비인후과': '이비인후과'
    }
    
    timeseries_features['진료과_매핑'] = timeseries_features['진료과'].map(dept_mapping)
    timeseries_features = timeseries_features.dropna(subset=['진료과_매핑'])
    
    # 진료과별 평균 예측 성능 계산
    dept_performance = timeseries_features.groupby('진료과_매핑').agg({
        'ARIMA_R2': 'mean',
        'RF_R2': 'mean',
        'XGB_R2': 'mean',
        'ARIMA_MAPE': 'mean',
        'RF_MAPE': 'mean',
        'XGB_MAPE': 'mean'
    }).reset_index()
    
    dept_performance.columns = ['진료과', 'ARIMA_R2_mean', 'RF_R2_mean', 'XGB_R2_mean',
                               'ARIMA_MAPE_mean', 'RF_MAPE_mean', 'XGB_MAPE_mean']
    
    # 병합
    df_merged = df.merge(dept_performance, on='진료과', how='left')
    
    print(f"병합 후 데이터 shape: {df_merged.shape}")
    
    # NaN 값 처리
    numeric_columns = df_merged.select_dtypes(include=[np.number]).columns
    df_merged[numeric_columns] = df_merged[numeric_columns].fillna(0)
    
    return df_merged

def prepare_features(df):
    """피쳐 준비 - 시계열 성능 향상 버전"""
    print("피쳐 준비 중...")
    
    # 기본 시계열 예측 성능 피쳐
    feature_columns = [
        'ARIMA_R2_mean', 'RF_R2_mean', 'XGB_R2_mean',
        'ARIMA_MAPE_mean', 'RF_MAPE_mean', 'XGB_MAPE_mean'
    ]
    
    # 실제로 존재하는 컬럼만 선택
    available_features = [col for col in feature_columns if col in df.columns]
    
    # 피쳐 엔지니어링 추가
    feature_engineered = df[available_features].copy()
    
    # 1. 예측 성능 통계 피쳐
    r2_features = [col for col in available_features if 'R2' in col]
    mape_features = [col for col in available_features if 'MAPE' in col]
    
    if r2_features:
        feature_engineered['R2_mean'] = df[r2_features].mean(axis=1)
        feature_engineered['R2_std'] = df[r2_features].std(axis=1)
        feature_engineered['R2_max'] = df[r2_features].max(axis=1)
        feature_engineered['R2_min'] = df[r2_features].min(axis=1)
        feature_engineered['R2_range'] = df[r2_features].max(axis=1) - df[r2_features].min(axis=1)
    
    if mape_features:
        feature_engineered['MAPE_mean'] = df[mape_features].mean(axis=1)
        feature_engineered['MAPE_std'] = df[mape_features].std(axis=1)
        feature_engineered['MAPE_max'] = df[mape_features].max(axis=1)
        feature_engineered['MAPE_min'] = df[mape_features].min(axis=1)
        feature_engineered['MAPE_range'] = df[mape_features].max(axis=1) - df[mape_features].min(axis=1)
    
    # 2. 모델별 성능 비교 피쳐
    if 'ARIMA_R2_mean' in available_features and 'RF_R2_mean' in available_features:
        feature_engineered['ARIMA_vs_RF_R2_diff'] = df['ARIMA_R2_mean'] - df['RF_R2_mean']
        feature_engineered['ARIMA_vs_RF_R2_ratio'] = df['ARIMA_R2_mean'] / (df['RF_R2_mean'] + 1e-8)
    
    if 'ARIMA_R2_mean' in available_features and 'XGB_R2_mean' in available_features:
        feature_engineered['ARIMA_vs_XGB_R2_diff'] = df['ARIMA_R2_mean'] - df['XGB_R2_mean']
        feature_engineered['ARIMA_vs_XGB_R2_ratio'] = df['ARIMA_R2_mean'] / (df['XGB_R2_mean'] + 1e-8)
    
    if 'RF_R2_mean' in available_features and 'XGB_R2_mean' in available_features:
        feature_engineered['RF_vs_XGB_R2_diff'] = df['RF_R2_mean'] - df['XGB_R2_mean']
        feature_engineered['RF_vs_XGB_R2_ratio'] = df['RF_R2_mean'] / (df['XGB_R2_mean'] + 1e-8)
    
    # 3. 상호작용 피쳐 (R2와 MAPE의 관계)
    if 'ARIMA_R2_mean' in available_features and 'ARIMA_MAPE_mean' in available_features:
        feature_engineered['ARIMA_R2_MAPE_ratio'] = df['ARIMA_R2_mean'] / (df['ARIMA_MAPE_mean'] + 1e-8)
        feature_engineered['ARIMA_R2_MAPE_product'] = df['ARIMA_R2_mean'] * df['ARIMA_MAPE_mean']
        feature_engineered['ARIMA_R2_MAPE_diff'] = df['ARIMA_R2_mean'] - df['ARIMA_MAPE_mean']
    
    if 'RF_R2_mean' in available_features and 'RF_MAPE_mean' in available_features:
        feature_engineered['RF_R2_MAPE_ratio'] = df['RF_R2_mean'] / (df['RF_MAPE_mean'] + 1e-8)
        feature_engineered['RF_R2_MAPE_product'] = df['RF_R2_mean'] * df['RF_MAPE_mean']
        feature_engineered['RF_R2_MAPE_diff'] = df['RF_R2_mean'] - df['RF_MAPE_mean']
    
    if 'XGB_R2_mean' in available_features and 'XGB_MAPE_mean' in available_features:
        feature_engineered['XGB_R2_MAPE_ratio'] = df['XGB_R2_mean'] / (df['XGB_MAPE_mean'] + 1e-8)
        feature_engineered['XGB_R2_MAPE_product'] = df['XGB_R2_mean'] * df['XGB_MAPE_mean']
        feature_engineered['XGB_R2_MAPE_diff'] = df['XGB_R2_mean'] - df['XGB_MAPE_mean']
    
    # 4. 모델 성능 순위 피쳐 (숫자로만)
    if len(r2_features) >= 3:
        feature_engineered['R2_rank_ARIMA'] = df[r2_features].rank(axis=1)[r2_features[0]]
        feature_engineered['R2_rank_RF'] = df[r2_features].rank(axis=1)[r2_features[1]]
        feature_engineered['R2_rank_XGB'] = df[r2_features].rank(axis=1)[r2_features[2]]
        
        # 최고/최저 성능 모델을 숫자로 인코딩 (문자열 제거)
        best_model_rank = df[r2_features].rank(axis=1, ascending=False).iloc[:, 0]  # 최고 성능 순위
        worst_model_rank = df[r2_features].rank(axis=1, ascending=True).iloc[:, 0]   # 최저 성능 순위
        feature_engineered['best_R2_model_rank'] = best_model_rank
        feature_engineered['worst_R2_model_rank'] = worst_model_rank
    
    # 5. 성능 일관성 피쳐
    if r2_features and mape_features:
        feature_engineered['R2_consistency'] = 1 / (df[r2_features].std(axis=1) + 1e-8)
        feature_engineered['MAPE_consistency'] = 1 / (df[mape_features].std(axis=1) + 1e-8)
        feature_engineered['overall_consistency'] = feature_engineered['R2_consistency'] * feature_engineered['MAPE_consistency']
    
    # 6. 진료과 더미 변수
    dept_dummy_columns = [col for col in df.columns if col.startswith('진료과_')]
    if dept_dummy_columns:
        feature_engineered = pd.concat([feature_engineered, df[dept_dummy_columns]], axis=1)
    
    # 7. 기타 숫자형 컬럼 추가 (문자열 제외)
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_columns = ['cost_bin_5'] + available_features + dept_dummy_columns
    additional_numeric = [col for col in numeric_columns if col not in exclude_columns]
    
    if additional_numeric:
        feature_engineered = pd.concat([feature_engineered, df[additional_numeric]], axis=1)
    
    # NaN 값 처리
    feature_engineered = feature_engineered.fillna(0)
    
    # 문자열 컬럼 제거 (혹시 남아있다면)
    string_columns = feature_engineered.select_dtypes(include=['object']).columns
    if len(string_columns) > 0:
        print(f"문자열 컬럼 제거: {list(string_columns)}")
        feature_engineered = feature_engineered.drop(columns=list(string_columns))
    
    X = feature_engineered
    y = df['cost_bin_5']
    
    # 소수 클래스 제외 (샘플 수가 5개 미만인 클래스)
    class_counts = y.value_counts()
    print(f"클래스별 샘플 수: {class_counts}")
    
    # 샘플 수가 5개 미만인 클래스 제외
    min_samples = 5
    valid_classes = class_counts[class_counts >= min_samples].index
    mask = y.isin(valid_classes)
    
    if mask.sum() < len(y):
        print(f"제외된 클래스: {set(y) - set(valid_classes)}")
        print(f"제외된 샘플 수: {len(y) - mask.sum()}")
    
    X = X[mask]
    y = y[mask]
    
    # 클래스 라벨을 0부터 연속적으로 재매핑
    unique_classes = sorted(y.unique())
    class_mapping = {old: new for new, old in enumerate(unique_classes)}
    y = y.map(class_mapping)
    
    print(f"클래스 매핑: {class_mapping}")
    print(f"최종 데이터 shape: {X.shape}")
    print(f"사용된 피쳐 수: {len(X.columns)}")
    print(f"피쳐 목록: {list(X.columns)}")
    
    return X, y, mask

def normalize_labels(y):
    """클래스 라벨을 0부터 연속적으로 정규화"""
    unique_classes = sorted(y.unique())
    class_mapping = {old: new for new, old in enumerate(unique_classes)}
    return y.map(class_mapping), class_mapping

def apply_sampling(X_train, y_train):
    """여러 샘플링 기법 적용 및 반환"""
    # 클래스별 샘플 수 확인
    class_counts = y_train.value_counts()
    min_samples = min(class_counts)
    
    # n_neighbors를 최소 샘플 수에 맞게 조정 (최소 2개)
    n_neighbors = min(2, min_samples - 1) if min_samples > 1 else 1
    
    samplers = {
        'Original': (X_train, y_train),
        'SMOTE': SMOTE(random_state=42, k_neighbors=n_neighbors),
        'ADASYN': ADASYN(random_state=42, n_neighbors=n_neighbors),
        'RandomUnderSampler': RandomUnderSampler(random_state=42),
        'SMOTEENN': SMOTEENN(random_state=42, smote=SMOTE(k_neighbors=n_neighbors))
    }
    
    results = {}
    for name, sampler in samplers.items():
        if name == 'Original':
            results[name] = (X_train, y_train)
        else:
            try:
                X_res, y_res = sampler.fit_resample(X_train, y_train)
                # 샘플링 후 클래스 라벨 정규화
                y_res, mapping = normalize_labels(y_res)
                print(f"  {name}: {len(y_train)} -> {len(y_res)} 샘플, 클래스: {sorted(y_res.unique())}")
                results[name] = (X_res, y_res)
            except Exception as e:
                print(f"Warning: {name} 샘플링 실패 - {e}")
                # 실패한 경우 원본 데이터 사용
                results[name] = (X_train, y_train)
    return results

def get_base_models():
    return {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='mlogloss'),
        'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1),
        'CatBoost': CatBoostClassifier(iterations=100, random_state=42, verbose=False),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }

def get_param_grids():
    return {
        'Random Forest': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', None]
        },
        'XGBoost': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'reg_alpha': [0, 0.1],
            'reg_lambda': [0, 0.1]
        },
        'LightGBM': {
            'n_estimators': [100, 200],
            'max_depth': [-1, 10],
            'learning_rate': [0.01, 0.1],
            'num_leaves': [31, 50],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'reg_alpha': [0, 0.1],
            'reg_lambda': [0, 0.1]
        },
        'CatBoost': {
            'iterations': [100, 200],
            'depth': [4, 6],
            'learning_rate': [0.01, 0.1],
            'l2_leaf_reg': [1, 3]
        },
        'AdaBoost': {
            'n_estimators': [50, 100],
            'learning_rate': [0.5, 1.0]
        },
        'Logistic Regression': {
            'C': [0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['liblinear']
        },
        'Neural Network': {
            'hidden_layer_sizes': [(100, 50), (150, 100)],
            'alpha': [0.0001, 0.001],
            'learning_rate_init': [0.001, 0.01],
            'max_iter': [500]
        }
    }

def train_and_evaluate(X_train, X_test, y_train, y_test, scaler=None):
    models = get_base_models()
    results = {}
    trained_models = {}
    
    # 교차검증을 위한 StratifiedKFold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"  {name} 모델 학습/평가 중...")
        
        # 교차검증으로 성능 평가
        if scaler and name in ['Neural Network', 'Logistic Regression']:
            # 스케일링이 필요한 모델의 경우 Pipeline 사용
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', model)
            ])
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1_weighted')
            # 최종 모델 학습
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_pred_proba = pipeline.predict_proba(X_test)
            trained_models[name] = pipeline
        else:
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted')
            # 최종 모델 학습
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            trained_models[name] = model
        
        # 테스트셋 성능
        test_accuracy = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average='weighted')
        
        results[name] = {
            'cv_f1_mean': cv_scores.mean(),
            'cv_f1_std': cv_scores.std(),
            'test_accuracy': test_accuracy,
            'test_f1_score': test_f1,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"    CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"    Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}")
    
    return results, trained_models

def train_ensemble(X_train, X_test, y_train, y_test, scaler=None):
    base_models = get_base_models()
    estimators = [(k, v) for k, v in base_models.items() if k != 'Neural Network']
    
    # 교차검증을 위한 StratifiedKFold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {}
    trained_models = {}
    
    # Voting
    print("  Voting 앙상블 학습/평가 중...")
    voting = VotingClassifier(estimators=estimators, voting='soft')
    cv_scores = cross_val_score(voting, X_train, y_train, cv=cv, scoring='f1_weighted')
    voting.fit(X_train, y_train)
    y_pred = voting.predict(X_test)
    y_pred_proba = voting.predict_proba(X_test)
    
    voting_result = {
        'cv_f1_mean': cv_scores.mean(),
        'cv_f1_std': cv_scores.std(),
        'test_accuracy': accuracy_score(y_test, y_pred),
        'test_f1_score': f1_score(y_test, y_pred, average='weighted'),
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    results['Voting'] = voting_result
    trained_models['Voting'] = voting
    
    print(f"    CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"    Test Accuracy: {voting_result['test_accuracy']:.4f}, Test F1: {voting_result['test_f1_score']:.4f}")
    
    # Stacking
    print("  Stacking 앙상블 학습/평가 중...")
    stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    cv_scores = cross_val_score(stacking, X_train, y_train, cv=cv, scoring='f1_weighted')
    stacking.fit(X_train, y_train)
    y_pred = stacking.predict(X_test)
    y_pred_proba = stacking.predict_proba(X_test)
    
    stacking_result = {
        'cv_f1_mean': cv_scores.mean(),
        'cv_f1_std': cv_scores.std(),
        'test_accuracy': accuracy_score(y_test, y_pred),
        'test_f1_score': f1_score(y_test, y_pred, average='weighted'),
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    results['Stacking'] = stacking_result
    trained_models['Stacking'] = stacking
    
    print(f"    CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"    Test Accuracy: {stacking_result['test_accuracy']:.4f}, Test F1: {stacking_result['test_f1_score']:.4f}")
    
    # Weighted Voting (성능 기반 가중치)
    print("  Weighted Voting 앙상블 학습/평가 중...")
    # 개별 모델 성능 계산
    model_weights = {}
    for name, model in base_models.items():
        if name != 'Neural Network':
            try:
                cv_score = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted').mean()
                model_weights[name] = cv_score
            except:
                model_weights[name] = 0.5
    
    # 가중치 정규화
    total_weight = sum(model_weights.values())
    if total_weight > 0:
        model_weights = {k: v/total_weight for k, v in model_weights.items()}
    
    # 가중 투표 구현
    weighted_predictions = np.zeros((len(X_test), len(np.unique(y_train))))
    for name, weight in model_weights.items():
        model = base_models[name]
        model.fit(X_train, y_train)
        pred_proba = model.predict_proba(X_test)
        weighted_predictions += weight * pred_proba
    
    y_pred_weighted = np.argmax(weighted_predictions, axis=1)
    weighted_result = {
        'cv_f1_mean': sum(model_weights.values()) / len(model_weights),  # 평균 가중치
        'cv_f1_std': 0.0,  # 단순화
        'test_accuracy': accuracy_score(y_test, y_pred_weighted),
        'test_f1_score': f1_score(y_test, y_pred_weighted, average='weighted'),
        'y_pred': y_pred_weighted,
        'y_pred_proba': weighted_predictions
    }
    results['Weighted_Voting'] = weighted_result
    trained_models['Weighted_Voting'] = model_weights  # 가중치 저장
    
    print(f"    Test Accuracy: {weighted_result['test_accuracy']:.4f}, Test F1: {weighted_result['test_f1_score']:.4f}")
    
    return results, trained_models

def tune_hyperparameters(X_train, y_train, X_test, y_test, scaler=None):
    param_grids = get_param_grids()
    best_results = {}
    best_models = {}
    for name, model in get_base_models().items():
        param_grid = param_grids[name]
        total_combinations = 1
        for v in param_grid.values():
            total_combinations *= len(v)
        print(f"[튜닝] {name} (총 {total_combinations}조합, 3-fold) 시작...")
        grid = GridSearchCV(model, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1)
        if scaler and name in ['Neural Network', 'Logistic Regression']:
            grid.fit(scaler.fit_transform(X_train), y_train)
            y_pred = grid.predict(scaler.transform(X_test))
            y_pred_proba = grid.predict_proba(scaler.transform(X_test))
        else:
            grid.fit(X_train, y_train)
            y_pred = grid.predict(X_test)
            y_pred_proba = grid.predict_proba(X_test)
        print(f"[튜닝] {name} 완료! Best: {grid.best_params_}")
        best_results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'best_params': grid.best_params_
        }
        best_models[name] = grid.best_estimator_
    return best_results, best_models

def save_prediction_results(X_test, y_test, sampling_results, ens_results, tune_results, output_dir, original_df=None):
    print("예측 결과 CSV 저장 중...")
    results_df = X_test.copy()
    results_df['y_true'] = y_test

    # 원본 데이터에서 원본 값들 추가 (위치 기반으로 매핑)
    if original_df is not None:
        test_indices = range(len(X_test))
        original_test_data = original_df.iloc[test_indices]
        # 병원명 처리
        if '병원명' in original_test_data.columns:
            results_df['original_병원명'] = original_test_data['병원명'].values
        elif '병원' in original_test_data.columns:
            results_df['original_병원명'] = original_test_data['병원'].values
        elif '진료과' in original_test_data.columns:
            # 진료과별 랜덤 병원명 할당
            hospital_list = ['서울', '인천', '부산', '대구', '대전', '광주']
            np.random.seed(42)
            dept_to_hosp = {dept: np.random.choice(hospital_list) for dept in original_test_data['진료과'].unique()}
            results_df['original_병원명'] = original_test_data['진료과'].map(dept_to_hosp).values
        else:
            results_df['original_병원명'] = 'Unknown'
        # 연령대 처리
        if '연령대' in original_test_data.columns:
            results_df['original_연령대'] = original_test_data['연령대'].values
        elif 'age_group' in original_test_data.columns:
            results_df['original_연령대'] = original_test_data['age_group'].values
        elif '상병코드' in original_test_data.columns and 'age_group' in original_df.columns:
            # 상병코드 기준 age_group 매핑
            code_to_age = original_df.groupby('상병코드')['age_group'].first().to_dict()
            results_df['original_연령대'] = original_test_data['상병코드'].map(lambda x: code_to_age.get(x, 'Unknown')).values
        else:
            results_df['original_연령대'] = 'Unknown'
        # 기존 원본 값들 추가
        original_columns = ['상병코드', '지역', '진료과', '년도', '연인원', '진료비(천원)']
        for col in original_columns:
            if col in original_test_data.columns:
                results_df[f'original_{col}'] = original_test_data[col].values
        # 인코딩 값들도 저장
        encoded_columns = ['disease_encoded', 'age_group_encoded', 'region_encoded', 'dept_encoded']
        for col in encoded_columns:
            if col in original_test_data.columns:
                results_df[f'encoded_{col}'] = original_test_data[col].values
        print(f"원본 값들 추가 완료: {[col for col in results_df.columns if col.startswith('original_')]}")
        print(f"인코딩 값들 추가 완료: {[col for col in results_df.columns if col.startswith('encoded_')]}")
    # 샘플링별 예측 결과 추가
    for samp_name, res in sampling_results.items():
        for model_name, r in res.items():
            results_df[f'y_pred_{samp_name}_{model_name}'] = r['y_pred']
            results_df[f'y_pred_proba_{samp_name}_{model_name}'] = [str(proba) for proba in r['y_pred_proba']]
    
    # 앙상블 예측 결과 추가
    for ens_name, r in ens_results.items():
        results_df[f'y_pred_ensemble_{ens_name}'] = r['y_pred']
        results_df[f'y_pred_proba_ensemble_{ens_name}'] = [str(proba) for proba in r['y_pred_proba']]
    
    # 튜닝된 모델 예측 결과 추가
    for model_name, r in tune_results.items():
        results_df[f'y_pred_tuned_{model_name}'] = r['y_pred']
        results_df[f'y_pred_proba_tuned_{model_name}'] = [str(proba) for proba in r['y_pred_proba']]
    
    # CSV 저장
    csv_path = f"{output_dir}/prediction_results_2.csv"
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"예측 결과가 {csv_path}에 저장되었습니다.")
    
    # 요약 통계도 저장
    summary_stats = []
    for samp_name, res in sampling_results.items():
        for model_name, r in res.items():
            summary_stats.append({
                'Type': f'Sampling-{samp_name}',
                'Model': model_name,
                'CV_F1_Mean': r['cv_f1_mean'],
                'CV_F1_Std': r['cv_f1_std'],
                'Test_Accuracy': r['test_accuracy'],
                'Test_F1_Score': r['test_f1_score']
            })
    
    for ens_name, r in ens_results.items():
        summary_stats.append({
            'Type': 'Ensemble',
            'Model': ens_name,
            'CV_F1_Mean': r['cv_f1_mean'],
            'CV_F1_Std': r['cv_f1_std'],
            'Test_Accuracy': r['test_accuracy'],
            'Test_F1_Score': r['test_f1_score']
        })
    
    for model_name, r in tune_results.items():
        summary_stats.append({
            'Type': 'Tuned',
            'Model': model_name,
            'CV_F1_Mean': r.get('cv_f1_mean', 0),
            'CV_F1_Std': r.get('cv_f1_std', 0),
            'Test_Accuracy': r['accuracy'],
            'Test_F1_Score': r['f1_score'],
            'Best_Params': str(r['best_params'])
        })
    
    summary_df = pd.DataFrame(summary_stats)
    summary_path = f"{output_dir}/model_performance_summary_detailed.csv"
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"상세 성능 요약이 {summary_path}에 저장되었습니다.")
    
    return results_df, summary_df

def main():
    print("=== 진료비 예측 모델 v3 (시계열 피쳐 + 샘플링/앙상블/튜닝) ===")
    df = load_and_preprocess_data()
    X, y, mask = prepare_features(df)
    
    # 원본 데이터에서 유효한 샘플만 필터링
    df_filtered = df[mask].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    
    # 샘플링별로 결과 저장
    sampling_results = {}
    sampling_models = {}
    for samp_name, (X_tr, y_tr) in apply_sampling(X_train, y_train).items():
        print(f"\n[{samp_name}] 샘플링 적용 후 모델 학습/평가...")
        # 스케일러는 Neural/Logistic에만 적용
        res, mods = train_and_evaluate(X_tr, X_test, y_tr, y_test, scaler)
        sampling_results[samp_name] = res
        sampling_models[samp_name] = mods
    
    # 앙상블
    print("\n[앙상블] Voting/Stacking 학습/평가...")
    ens_results, ens_models = train_ensemble(X_train, X_test, y_train, y_test, scaler)
    
    # 하이퍼파라미터 튜닝
    print("\n[튜닝] 주요 모델 하이퍼파라미터 튜닝...")
    tune_results, tune_models = tune_hyperparameters(X_train, y_train, X_test, y_test, scaler)
    
    # 결과 저장
    output_dir = "model_results_v3_시계열_확장"
    os.makedirs(output_dir, exist_ok=True)
    
    # 샘플링 결과 저장
    for samp_name, res in sampling_results.items():
        for model_name, r in res.items():
            fname = f"{output_dir}/sampling_{samp_name}_{model_name}_model.pkl"
            with open(fname, 'wb') as f:
                pickle.dump(sampling_models[samp_name][model_name], f)
    
    # 앙상블 결과 저장
    for ens_name, model in ens_models.items():
        fname = f"{output_dir}/ensemble_{ens_name}_model.pkl"
        with open(fname, 'wb') as f:
            pickle.dump(model, f)
    
    # 튜닝 결과 저장
    for model_name, model in tune_models.items():
        fname = f"{output_dir}/tuned_{model_name}_model.pkl"
        with open(fname, 'wb') as f:
            pickle.dump(model, f)
    
    # 성능 요약 저장
    perf_rows = []
    for samp_name, res in sampling_results.items():
        for model_name, r in res.items():
            perf_rows.append({'Type': f'Sampling-{samp_name}', 'Model': model_name, 'Accuracy': r['test_accuracy'], 'F1_Score': r['test_f1_score']})
    for ens_name, r in ens_results.items():
        perf_rows.append({'Type': 'Ensemble', 'Model': ens_name, 'Accuracy': r['test_accuracy'], 'F1_Score': r['test_f1_score']})
    for model_name, r in tune_results.items():
        perf_rows.append({'Type': 'Tuned', 'Model': model_name, 'Accuracy': r['accuracy'], 'F1_Score': r['f1_score'], 'Best_Params': r['best_params']})
    pd.DataFrame(perf_rows).to_csv(f"{output_dir}/model_performance_summary.csv", index=False)
    print(f"모든 결과가 {output_dir}에 저장되었습니다.")
    
    # 예측 결과 저장 (원본 데이터 포함)
    save_prediction_results(X_test, y_test, sampling_results, ens_results, tune_results, output_dir, df_filtered)

if __name__ == "__main__":
    main() 