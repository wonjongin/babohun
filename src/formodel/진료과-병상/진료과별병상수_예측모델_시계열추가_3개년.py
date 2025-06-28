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
from sklearn.impute import SimpleImputer
import os

# ----------------------
# 데이터 불러오기
# ----------------------
data = pd.read_csv("new_merged_data/병원_통합_데이터_호스피스 삭제.csv")
df_pred = pd.read_csv("analysis_data/병원별_진료과별_입원_미래3년_예측결과.csv")

# ----------------------
# df_pred 병원별 예측값 합산 (병상수 예측용으로 집계)
# ----------------------
df_pred_grouped = df_pred.groupby("병원").agg({
    "ARIMA예측": "sum",
    "RF예측": "sum",
    "XGB예측": "sum"
}).reset_index()

# ----------------------
# data와 병원명 기준 조인
# ----------------------
data = data.merge(df_pred_grouped, left_on="병원명", right_on="병원", how="left")

# ----------------------
# 가중예측값 계산
# ----------------------
for col in ['ARIMA예측', 'RF예측', 'XGB예측']:
    if col not in data.columns:
        raise ValueError(f"'{col}' 컬럼이 데이터에 없습니다.")

data['가중예측값'] = (
    0.2 * data['ARIMA예측'] +
    0.3 * data['RF예측'] +
    0.5 * data['XGB예측']
)

# ----------------------
# 병상수 컬럼 및 총병상수 생성
# ----------------------
bed_columns = [
    "격리병실", "무균치료실", "물리치료실", "분만실", "수술실",
    "신생아실", "응급실", "인공신장실", "일반입원실_상급", "일반입원실_일반",
    "정신과개방_일반", "정신과폐쇄_일반", "중환자실_성인", "회복실"
]
data["총병상수"] = data[bed_columns].sum(axis=1)

# ----------------------
# 타겟 및 독립변수 분리
# ----------------------

# 타겟: 병상수 다중 출력
y = data[bed_columns + ["총병상수"]]

# 독립변수: 가중예측값 포함, 병원명 원-핫 인코딩
X = data.drop(columns=bed_columns + ["총병상수", "병원", "병원명"])  # 조인 후 생긴 중복 병원명 컬럼 제외
X = pd.get_dummies(X.join(data["병원명"]), columns=["병원명"], drop_first=False)

# 결측치 처리 및 스케일링
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# ----------------------
# 교차검증 설정
# ----------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# ----------------------
# 모델 및 하이퍼파라미터 그리드
# ----------------------
models_and_params = {
    "RandomForest": {
        "model": RandomForestRegressor(random_state=42),
        "params": {
            "estimator__n_estimators": [50, 100],
            "estimator__max_depth": [None, 10, 20]
        }
    },
    "GradientBoosting": {
        "model": GradientBoostingRegressor(random_state=42),
        "params": {
            "estimator__n_estimators": [50, 100],
            "estimator__learning_rate": [0.05, 0.1],
            "estimator__max_depth": [3, 5]
        }
    },
    "Ridge": {
        "model": Ridge(),
        "params": {
            "estimator__alpha": [0.1, 1.0, 10.0]
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
            "estimator__max_depth": [None, 10, 20],
            "estimator__min_samples_split": [2, 5]
        }
    },
    "AdaBoost": {
        "model": AdaBoostRegressor(random_state=42),
        "params": {
            "estimator__n_estimators": [50, 100],
            "estimator__learning_rate": [0.05, 0.1]
        }
    },
    "KNN": {
        "model": KNeighborsRegressor(),
        "params": {
            "estimator__n_neighbors": [3, 5, 7],
            "estimator__weights": ['uniform', 'distance']
        }
    }
}

# ----------------------
# 모델 학습 및 평가
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

        pred_df = y.copy()

        # 1. 병원명 칼럼 추가 (원본 data의 병원명 열 복사)
        pred_df["병원명"] = data["병원명"].values

        # 2. 예측 결과 칼럼 추가
        for i, col in enumerate(y.columns):
            pred_df[f"{col}_예측_{name}"] = y_pred[:, i]

        # 3. 병원명 칼럼을 맨 앞으로 이동 (선택사항)
        cols = pred_df.columns.tolist()
        cols = ['병원명'] + [c for c in cols if c != '병원명']
        pred_df = pred_df[cols]

        print(pred_df)
        pred_dfs[name] = pred_df

    except Exception as e:
        print(f"{name} 모델 처리 중 오류 발생: {e}")
        continue

# ----------------------
# 결과 저장
# ----------------------

# 결과 저장 디렉토리 생성
results_dir = "model_results_진료과별병상수_예측모델_시계열추가_3개년"
os.makedirs(results_dir, exist_ok=True)

results_df = pd.DataFrame(results)
results_df.to_csv(f"{results_dir}/hospital_bed_model_comparison_metrics_gridcv.csv", encoding="utf-8-sig", index=False)
print(results_df)

for name, pred_df in pred_dfs.items():
    pred_df.to_csv(f"{results_dir}/hospital_bed_prediction_results_{name}_gridcv.csv", encoding="utf-8-sig", index=False)
