import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import os

# ----------------------
# 데이터 불러오기 및 전처리
# ----------------------
print("=== 데이터 로딩 시작 ===")

# 기본 데이터
data = pd.read_csv("new_merged_data/병원_통합_데이터.csv")
data = data[~data["병원명"].str.contains("호스피스")]  # 호스피스 병원 제외

# 연령지역 데이터 로드 (성능 향상을 위해)
try:
    df_age_region = pd.read_csv('model_results_연령지역_진료과/Stacking_prediction_results_detailed.csv')
    print("✅ 연령지역 진료과 데이터 로드 완료")
except:
    print("⚠️ 연령지역 진료과 데이터 없음")
    df_age_region = None

# 시계열 예측 데이터 로드 (가능한 경우)
# try:
#     df_pred = pd.read_csv('analysis_data/병원별_진료과별_입원외래_통합_시계열예측결과_개선.csv')
#     print("✅ 시계열 예측 데이터 로드 완료")
# except:
#     print("⚠️ 시계열 예측 데이터 없음")
df_pred = None

print("=== 데이터 전처리 시작 ===")

bed_columns = [
    "격리병실", "무균치료실", "물리치료실", "분만실", "수술실",
    "신생아실", "응급실", "인공신장실", "일반입원실_상급", "일반입원실_일반",
    "정신과개방_일반", "정신과폐쇄_일반", "중환자실_성인", "회복실"
]

data["총병상수"] = data[bed_columns].sum(axis=1)
y = data[bed_columns + ["총병상수"]]

# 병원명 정보를 별도로 저장
hospital_names = data["병원명"].copy()

drop_cols = bed_columns + ["총병상수", "병원명"]
X = data.drop(columns=drop_cols)

# ----------------------
# 연령지역 데이터 기반 추가 피처 생성
# ----------------------
print("=== 연령지역 데이터 기반 피처 생성 ===")

# 1) 지역 정보 추출 (병원명에서)
X['지역'] = hospital_names.apply(lambda x: 
    '서울' if '서울' in str(x) else
    '광역시' if any(city in str(x) for city in ['부산', '대구', '인천', '광주', '대전']) else
    '기타'
)

X['대도시'] = X['지역'].apply(lambda x: 1 if x in ['서울', '광역시'] else 0)

# 2) 병원 규모 분류
X['병원규모'] = data['총병상수'].apply(lambda x: 
    '대형' if x >= 1000 else
    '중형' if x >= 500 else
    '소형'
)

# 3) 연령지역 데이터에서 진료과별 인기도 정보 추가
if df_age_region is not None:
    # 진료과별 인기도 계산
    department_popularity = df_age_region.groupby('y_actual').agg({
        'top1_probability': 'mean',
        'confidence': 'mean',
        'sample_weight': 'sum'
    }).reset_index()
    
    department_popularity.columns = ['진료과', '평균확률', '평균신뢰도', '총샘플수']
    
    # 병원별 진료과 정보가 있다면 매핑
    if '진료과' in X.columns:
        X = X.merge(department_popularity, on='진료과', how='left')
        X['평균확률'] = X['평균확률'].fillna(0)
        X['평균신뢰도'] = X['평균신뢰도'].fillna(0)
        X['총샘플수'] = X['총샘플수'].fillna(0)
    else:
        # 진료과 정보가 없으면 전체 평균값 사용
        X['평균확률'] = department_popularity['평균확률'].mean()
        X['평균신뢰도'] = department_popularity['평균신뢰도'].mean()
        X['총샘플수'] = department_popularity['총샘플수'].mean()
else:
    X['평균확률'] = 0
    X['평균신뢰도'] = 0
    X['총샘플수'] = 0

# 4) 시계열 예측 데이터 추가 (가능한 경우)
if df_pred is not None:
    # 병원별 예측 데이터 통합
    pred_summary = df_pred.groupby('병원').agg({
        'ARIMA예측': 'mean',
        'RF예측': 'mean',
        'XGB예측': 'mean',
        '실제값': 'mean'
    }).reset_index()
    
    pred_summary.columns = ['병원명', 'ARIMA예측_평균', 'RF예측_평균', 'XGB예측_평균', '실제값_평균']
    
    # 병원명을 인덱스로 설정하여 병합
    X_with_names = X.copy()
    X_with_names['병원명'] = hospital_names
    X_with_names = X_with_names.merge(pred_summary, on='병원명', how='left')
    X = X_with_names.drop(columns=['병원명'])
    
    # 예측값 관련 피처 생성
    X['예측값_평균'] = X[['ARIMA예측_평균', 'RF예측_평균', 'XGB예측_평균']].mean(axis=1)
    X['예측값_표준편차'] = X[['ARIMA예측_평균', 'RF예측_평균', 'XGB예측_평균']].std(axis=1)
    X['가중예측값'] = (0.2 * X['ARIMA예측_평균'] + 0.3 * X['RF예측_평균'] + 0.5 * X['XGB예측_평균'])
    
    # NaN 값 처리
    X = X.fillna(0)
else:
    # 예측 데이터가 없는 경우 기본값 설정
    X['ARIMA예측_평균'] = 0
    X['RF예측_평균'] = 0
    X['XGB예측_평균'] = 0
    X['실제값_평균'] = 0
    X['예측값_평균'] = 0
    X['예측값_표준편차'] = 0
    X['가중예측값'] = 0

# 5) 상호작용 피처 생성
X['총병상수_대도시'] = data['총병상수'] * X['대도시']
X['총병상수_평균확률'] = data['총병상수'] * X['평균확률']
X['대도시_평균신뢰도'] = X['대도시'] * X['평균신뢰도']

# 6) 다항식 피처
X['총병상수_제곱'] = data['총병상수'] ** 2
X['총병상수_세제곱'] = data['총병상수'] ** 3

# 7) 로그 변환
X['총병상수_log'] = np.log1p(data['총병상수'])
X['평균확률_log'] = np.log1p(np.abs(X['평균확률']))
X['총샘플수_log'] = np.log1p(X['총샘플수'])

print(f"기존 피처 수: {len(X.columns) - len(bed_columns) - 1}개")
print(f"추가된 피처 수: {len([col for col in X.columns if col not in data.columns])}개")
print(f"총 피처 수: {len(X.columns)}개")

# 원핫 인코딩
X = pd.get_dummies(X, drop_first=True)
X.fillna(X.mean(), inplace=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"최종 피처 수: {X_scaled.shape[1]}개")

# ----------------------
# 교차검증용 KFold 설정
# ----------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# ----------------------
# 모델 및 하이퍼파라미터 그리드 정의 (개선된 버전)
# ----------------------
models_and_params = {
    "RandomForest": {
        "model": RandomForestRegressor(random_state=42),
        "params": {
            "estimator__n_estimators": [100, 200],
            "estimator__max_depth": [None, 10, 15],
            "estimator__min_samples_split": [2, 5, 10]
        }
    },
    "GradientBoosting": {
        "model": GradientBoostingRegressor(random_state=42),
        "params": {
            "estimator__n_estimators": [100, 200],
            "estimator__learning_rate": [0.05, 0.1, 0.15],
            "estimator__max_depth": [3, 5, 7]
        }
    },
    "Ridge": {
        "model": Ridge(),
        "params": {
            "estimator__alpha": [0.1, 1.0, 10.0, 100.0]
        }
    },
    "ElasticNet": {
        "model": ElasticNet(random_state=42),
        "params": {
            "estimator__alpha": [0.1, 1.0, 10.0],
            "estimator__l1_ratio": [0.2, 0.5, 0.8]
        }
    },
    "DecisionTree": {
        "model": DecisionTreeRegressor(random_state=42),
        "params": {
            "estimator__max_depth": [None, 10, 15, 20],
            "estimator__min_samples_split": [2, 5, 10],
            "estimator__min_samples_leaf": [1, 2, 4]
        }
    },
    "AdaBoost": {
        "model": AdaBoostRegressor(random_state=42),
        "params": {
            "estimator__n_estimators": [50, 100, 200],
            "estimator__learning_rate": [0.05, 0.1, 0.15]
        }
    },
    "KNN": {
        "model": KNeighborsRegressor(),
        "params": {
            "estimator__n_neighbors": [3, 5, 7, 9],
            "estimator__weights": ['uniform', 'distance']
        }
    }
}

# ----------------------
# 결과 저장 디렉토리 생성
# ----------------------
results_dir = "model_results_진료과별병상수_예측모델_연령지역진료과추가"
os.makedirs(results_dir, exist_ok=True)

print(f"📁 결과 저장 디렉토리: {results_dir}/")

# ----------------------
# 그리드 서치 + 교차 검증 + 평가
# ----------------------
results = []
pred_dfs = {}

for name, mp in models_and_params.items():
    print(f"### {name} 모델 그리드 서치 및 교차검증 시작 ###")
    base_model = MultiOutputRegressor(mp["model"])
    grid = GridSearchCV(
        estimator=base_model,
        param_grid=mp["params"],
        cv=kf,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    try:
        grid.fit(X_scaled, y)
        best_model = grid.best_estimator_
        print(f"Best params for {name}: {grid.best_params_}")
        print(f"Best CV MSE (neg): {grid.best_score_}")

        # 전체 데이터로 예측
        y_pred = best_model.predict(X_scaled)

        for i, col in enumerate(y.columns):
            true_vals = y[col]
            pred_vals = y_pred[:, i]

            mse = mean_squared_error(true_vals, pred_vals)
            mae = mean_absolute_error(true_vals, pred_vals)
            r2 = r2_score(true_vals, pred_vals)

            results.append({
                "모델": name,
                "병상종류": col,
                "MSE": mse,
                "MAE": mae,
                "R2": r2
            })

        pred_df = y.copy()
        for i, col in enumerate(y.columns):
            pred_df[f"{col}_예측_{name}"] = y_pred[:, i]
        pred_dfs[name] = pred_df

    except Exception as e:
        print(f"{name} 모델 처리 중 오류 발생: {e}")
        continue

# ----------------------
# 결과 저장 및 출력
# ----------------------
print("\n=== 결과 저장 시작 ===")

# 성능 비교 결과 저장
results_df = pd.DataFrame(results)
results_df.to_csv(f"{results_dir}/hospital_bed_model_comparison_metrics_gridcv.csv", encoding="utf-8-sig", index=False)
print(results_df)

# 개별 모델 예측 결과 저장
for name, pred_df in pred_dfs.items():
    pred_df.to_csv(f"{results_dir}/hospital_bed_prediction_results_{name}_gridcv.csv", encoding="utf-8-sig", index=True)
    print(f"✅ {name} 예측 결과 저장 완료")

print(f"\n🎉 모든 결과가 {results_dir}/ 디렉토리에 저장되었습니다!")
print("📊 성능 비교: hospital_bed_model_comparison_metrics_gridcv.csv")
print("🎯 예측 결과: hospital_bed_prediction_results_*.csv")
