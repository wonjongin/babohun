import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# XGBWrapper 클래스 정의 (모델 로드에 필요)
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

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

print("=== Stacking 모델 예측 결과 출력 ===")
print("="*50)

# 1. 데이터 로드
print("1/5: 데이터 로드 중...")
df = pd.read_csv("new_merged_data/df_result2_with_심평원.csv", dtype=str)
age_cols = ["59이하", "60-64", "65-69", "70-79", "80-89", "90이상"]

# 2. 데이터 전처리 (원본 코드와 동일)
print("2/5: 데이터 전처리 중...")
m = df.melt(
    id_vars=["년도", "구분", "지역", "상병코드", "진료과"],
    value_vars=age_cols,
    var_name="age_group",
    value_name="count",
)
m["count"] = pd.to_numeric(m["count"], errors="coerce").fillna(0).astype(int)
m["대표진료과"] = m["진료과"]
train = m[m["대표진료과"].notna()]

# 강화된 피처 엔지니어링 (원본과 동일)
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

print(f"데이터 크기: {X.shape}")
print(f"클래스 수: {len(np.unique(y))}")

# 3. Stacking 모델 로드
print("3/5: Stacking 모델 로드 중...")
try:
    stacking_model = joblib.load("model_results_연령지역_진료과/models/Stacking_model.pkl")
    print("✅ Stacking 모델 로드 완료")
except Exception as e:
    print(f"❌ Stacking 모델 로드 실패: {str(e)}")
    exit()

# 4. 예측 수행
print("4/5: 예측 수행 중...")
try:
    # 예측
    y_pred = stacking_model.predict(X)
    y_proba = stacking_model.predict_proba(X)
    
    print("✅ 예측 완료")
    print(f"예측된 클래스 수: {len(np.unique(y_pred))}")
    
except Exception as e:
    print(f"❌ 예측 실패: {str(e)}")
    exit()

# 5. 결과를 DataFrame으로 변환
print("5/5: 결과 DataFrame 생성 중...")

# 예측 결과 DataFrame 생성
result_df = X.copy()

# 예측 결과 추가
result_df['y_actual'] = y.values
result_df['y_predicted'] = y_pred
result_df['prediction_correct'] = (y.values == y_pred).astype(int)

# 예측 확률 추가 (상위 3개 클래스)
if hasattr(stacking_model, 'classes_'):
    class_names = stacking_model.classes_
elif hasattr(stacking_model, 'orig_classes_'):
    class_names = stacking_model.orig_classes_
else:
    class_names = [f"class_{i}" for i in range(y_proba.shape[1])]

# 상위 3개 예측 확률 추가
top3_indices = np.argsort(y_proba, axis=1)[:, -3:][:, ::-1]
top3_classes = class_names[top3_indices]
top3_probs = np.take_along_axis(y_proba, top3_indices, axis=1)

result_df['top1_class'] = top3_classes[:, 0]
result_df['top1_probability'] = top3_probs[:, 0]
result_df['top2_class'] = top3_classes[:, 1]
result_df['top2_probability'] = top3_probs[:, 1]
result_df['top3_class'] = top3_classes[:, 2]
result_df['top3_probability'] = top3_probs[:, 2]

# 신뢰도 (최대 확률값)
result_df['confidence'] = y_proba.max(axis=1)

# 가중치 정보 추가
result_df['sample_weight'] = w.values

# 6. 결과 저장
print("\n=== 결과 저장 ===")

# CSV 파일로 저장
output_file = "model_results_연령지역_진료과/Stacking_prediction_results_detailed.csv"
result_df.to_csv(output_file, encoding='utf-8-sig', index=False)

print(f"✅ 상세 예측 결과 저장 완료: {output_file}")
print(f"📊 총 {len(result_df)}개 예측 결과")

# 7. 성능 요약 출력
print("\n=== 성능 요약 ===")
from sklearn.metrics import accuracy_score, classification_report

# 전체 성능
overall_accuracy = accuracy_score(y, y_pred, sample_weight=w)
print(f"전체 정확도 (가중치 적용): {overall_accuracy:.4f}")

# 가중치 적용하지 않은 정확도
simple_accuracy = accuracy_score(y, y_pred)
print(f"전체 정확도 (가중치 미적용): {simple_accuracy:.4f}")

# 예측 신뢰도 통계
print(f"\n예측 신뢰도 통계:")
print(f"평균 신뢰도: {result_df['confidence'].mean():.4f}")
print(f"신뢰도 표준편차: {result_df['confidence'].std():.4f}")
print(f"최소 신뢰도: {result_df['confidence'].min():.4f}")
print(f"최대 신뢰도: {result_df['confidence'].max():.4f}")

# 상위 예측 결과
print(f"\n상위 예측 결과:")
print(f"Top-1 정확도: {(result_df['y_actual'] == result_df['top1_class']).mean():.4f}")
print(f"Top-2 정확도: {((result_df['y_actual'] == result_df['top1_class']) | (result_df['y_actual'] == result_df['top2_class'])).mean():.4f}")
print(f"Top-3 정확도: {((result_df['y_actual'] == result_df['top1_class']) | (result_df['y_actual'] == result_df['top2_class']) | (result_df['y_actual'] == result_df['top3_class'])).mean():.4f}")

# 클래스별 성능
print(f"\n클래스별 성능 (상위 10개 클래스):")
class_performance = result_df.groupby('y_actual').agg({
    'prediction_correct': 'mean',
    'confidence': 'mean',
    'sample_weight': 'sum'
}).sort_values('sample_weight', ascending=False).head(10)

print(class_performance.round(4))

# 8. 추가 분석 파일 생성
print("\n=== 추가 분석 파일 생성 ===")

# 신뢰도 구간별 성능 분석
confidence_bins = [0, 0.5, 0.7, 0.8, 0.9, 1.0]
confidence_labels = ['0-0.5', '0.5-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
result_df['confidence_bin'] = pd.cut(result_df['confidence'], bins=confidence_bins, labels=confidence_labels)

confidence_performance = result_df.groupby('confidence_bin').agg({
    'prediction_correct': 'mean',
    'sample_weight': 'sum'
}).reset_index()
confidence_performance.columns = ['confidence_bin', 'accuracy', 'total_weight']

confidence_file = "model_results_연령지역_진료과/Stacking_confidence_analysis.csv"
confidence_performance.to_csv(confidence_file, encoding='utf-8-sig', index=False)
print(f"✅ 신뢰도 분석 저장: {confidence_file}")

# 오분류 분석
misclassified = result_df[result_df['prediction_correct'] == 0].copy()
if len(misclassified) > 0:
    misclassified_file = "model_results_연령지역_진료과/Stacking_misclassified_cases.csv"
    misclassified.to_csv(misclassified_file, encoding='utf-8-sig', index=False)
    print(f"✅ 오분류 케이스 저장: {misclassified_file}")
    print(f"📊 오분류 케이스 수: {len(misclassified)}개")

print("\n" + "="*50)
print("🎉 Stacking 모델 예측 결과 출력 완료!")
print("="*50)
print(f"📁 주요 결과 파일:")
print(f"  - 상세 예측 결과: {output_file}")
print(f"  - 신뢰도 분석: {confidence_file}")
if len(misclassified) > 0:
    print(f"  - 오분류 케이스: {misclassified_file}")
print("="*50)
