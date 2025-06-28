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

# ----------------------
# 데이터 불러오기 및 전처리
# ----------------------
data = pd.read_csv("new_merged_data/병원_통합_데이터.csv")
data = data[~data["병원명"].str.contains("호스피스")]  # 호스피스 병원 제외

bed_columns = [
    "격리병실", "무균치료실", "물리치료실", "분만실", "수술실",
    "신생아실", "응급실", "인공신장실", "일반입원실_상급", "일반입원실_일반",
    "정신과개방_일반", "정신과폐쇄_일반", "중환자실_성인", "회복실"
]

data["총병상수"] = data[bed_columns].sum(axis=1)
y = data[bed_columns + ["총병상수"]]
drop_cols = bed_columns + ["총병상수", "병원명"]
X = data.drop(columns=drop_cols)

X = pd.get_dummies(X, drop_first=True)
X.fillna(X.mean(), inplace=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------
# 교차검증용 KFold 설정
# ----------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# ----------------------
# 모델 및 하이퍼파라미터 그리드 정의
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

        # 테스트셋 따로 나눠서 평가하려면 아래 주석 해제하고 사용 가능
        # X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        # best_model.fit(X_train, y_train)
        # y_pred = best_model.predict(X_test)

        # 여기서는 전체 데이터로 CV 평가 기준으로 성능 확인
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
results_df = pd.DataFrame(results)
results_df.to_csv("bed_model_comparison_metrics_gridcv.csv", encoding="utf-8-sig", index=False)
print(results_df)

for name, pred_df in pred_dfs.items():
    pred_df.to_csv(f"bed_prediction_results_{name}_gridcv.csv", encoding="utf-8-sig", index=True)
