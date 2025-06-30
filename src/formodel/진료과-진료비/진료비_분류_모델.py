# -*- coding: utf-8 -*-
"""
진료비 구간 분류 모델
상병코드/진료과 기반 진료비 구간 예측
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Pretendard'
plt.rcParams['axes.unicode_minus'] = False

# ----------------------------------------------------------------------
# 1) 데이터 로드 및 전처리
# ----------------------------------------------------------------------
print("=== 데이터 로드 및 전처리 ===")

# 구간 정보가 포함된 데이터 로드
df = pd.read_csv("new_merged_data/df_result2_with_심평원_진료비_구간.csv", encoding="utf-8-sig")

print(f"전체 데이터: {len(df)}개")
print(f"컬럼: {list(df.columns)}")

# 진료비가 있는 데이터만 사용
df_with_cost = df[df['진료비(천원)'].notna()].copy()
print(f"진료비가 있는 데이터: {len(df_with_cost)}개")

# ----------------------------------------------------------------------
# 2) 특징 엔지니어링
# ----------------------------------------------------------------------
print("\n=== 특징 엔지니어링 ===")

# 연령대별 비율 계산
age_columns = ['59이하', '60-64', '65-69', '70-79', '80-89', '90이상']
for col in age_columns:
    df_with_cost[f'{col}_ratio'] = df_with_cost[col] / df_with_cost['연인원']

# 연령대별 비율의 표준편차 (연령 분포의 다양성)
df_with_cost['age_diversity'] = df_with_cost[age_columns].std(axis=1)

# 연령대별 비율의 최대값 (주요 연령대)
df_with_cost['main_age_group'] = df_with_cost[age_columns].idxmax(axis=1)

# 구분을 이진 변수로 변환
df_with_cost['is_inpatient'] = (df_with_cost['구분'] == '입원(연인원)').astype(int)

# 연도 정보 (2021=0, 2022=1, 2023=2)
df_with_cost['year_encoded'] = df_with_cost['년도'] - 2021

print("특징 엔지니어링 완료")

# ----------------------------------------------------------------------
# 3) 모델 A: 상병코드 기반 진료비 구간 예측
# ----------------------------------------------------------------------
print("\n=== 모델 A: 상병코드 기반 진료비 구간 예측 ===")

# 상병코드 인코딩
le_disease = LabelEncoder()
df_with_cost['disease_encoded'] = le_disease.fit_transform(df_with_cost['상병코드'])

# 특징 선택
features_A = [
    'disease_encoded', 'year_encoded', 'is_inpatient', '연인원',
    '59이하_ratio', '60-64_ratio', '65-69_ratio', '70-79_ratio', '80-89_ratio', '90이상_ratio',
    'age_diversity'
]

X_A = df_with_cost[features_A]
y_A = df_with_cost['cost_bin']

print(f"상병코드 기반 모델 특징: {len(features_A)}개")
print(f"목표 변수 분포:\n{y_A.value_counts().sort_index()}")

# 데이터 분할
X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(
    X_A, y_A, test_size=0.2, random_state=42, stratify=y_A
)

# 상병코드 기반 모델 학습
print("\n상병코드 기반 모델 학습 중...")
rf_model_A = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model_A.fit(X_train_A, y_train_A)

# 예측 및 평가
y_pred_A = rf_model_A.predict(X_test_A)
accuracy_A = accuracy_score(y_test_A, y_pred_A)

print(f"\n상병코드 기반 모델 성능:")
print(f"정확도: {accuracy_A:.4f}")

# 교차 검증
cv_scores_A = cross_val_score(rf_model_A, X_A, y_A, cv=5, scoring='accuracy')
print(f"교차 검증 정확도: {cv_scores_A.mean():.4f} (+/- {cv_scores_A.std() * 2:.4f})")

# 분류 보고서
print("\n상병코드 기반 모델 분류 보고서:")
print(classification_report(y_test_A, y_pred_A))

# ----------------------------------------------------------------------
# 4) 모델 B: 진료과 기반 진료비 구간 예측
# ----------------------------------------------------------------------
print("\n=== 모델 B: 진료과 기반 진료비 구간 예측 ===")

# 진료과 인코딩
le_dept = LabelEncoder()
df_with_cost['dept_encoded'] = le_dept.fit_transform(df_with_cost['진료과'])

# 특징 선택
features_B = [
    'dept_encoded', 'year_encoded', 'is_inpatient', '연인원',
    '59이하_ratio', '60-64_ratio', '65-69_ratio', '70-79_ratio', '80-89_ratio', '90이상_ratio',
    'age_diversity'
]

X_B = df_with_cost[features_B]
y_B = df_with_cost['cost_bin']

print(f"진료과 기반 모델 특징: {len(features_B)}개")

# 데이터 분할
X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(
    X_B, y_B, test_size=0.2, random_state=42, stratify=y_B
)

# 진료과 기반 모델 학습
print("\n진료과 기반 모델 학습 중...")
rf_model_B = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model_B.fit(X_train_B, y_train_B)

# 예측 및 평가
y_pred_B = rf_model_B.predict(X_test_B)
accuracy_B = accuracy_score(y_test_B, y_pred_B)

print(f"\n진료과 기반 모델 성능:")
print(f"정확도: {accuracy_B:.4f}")

# 교차 검증
cv_scores_B = cross_val_score(rf_model_B, X_B, y_B, cv=5, scoring='accuracy')
print(f"교차 검증 정확도: {cv_scores_B.mean():.4f} (+/- {cv_scores_B.std() * 2:.4f})")

# 분류 보고서
print("\n진료과 기반 모델 분류 보고서:")
print(classification_report(y_test_B, y_pred_B))

# ----------------------------------------------------------------------
# 5) 모델 비교 및 시각화
# ----------------------------------------------------------------------
print("\n=== 모델 비교 ===")

# 혼동 행렬 시각화
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 상병코드 기반 모델 혼동 행렬
cm_A = confusion_matrix(y_test_A, y_pred_A)
sns.heatmap(cm_A, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('상병코드 기반 모델 혼동 행렬')
axes[0].set_xlabel('예측 구간')
axes[0].set_ylabel('실제 구간')

# 진료과 기반 모델 혼동 행렬
cm_B = confusion_matrix(y_test_B, y_pred_B)
sns.heatmap(cm_B, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('진료과 기반 모델 혼동 행렬')
axes[1].set_xlabel('예측 구간')
axes[1].set_ylabel('실제 구간')

plt.tight_layout()
plt.savefig('new_merged_data/모델_혼동행렬.png', dpi=300, bbox_inches='tight')
plt.show()

# 모델 성능 비교
performance_comparison = pd.DataFrame({
    '모델': ['상병코드 기반', '진료과 기반'],
    '정확도': [accuracy_A, accuracy_B],
    '교차검증_평균': [cv_scores_A.mean(), cv_scores_B.mean()],
    '교차검증_표준편차': [cv_scores_A.std(), cv_scores_B.std()]
})

print("\n모델 성능 비교:")
print(performance_comparison)

# ----------------------------------------------------------------------
# 6) 특징 중요도 분석
# ----------------------------------------------------------------------
print("\n=== 특징 중요도 분석 ===")

# 상병코드 기반 모델 특징 중요도
feature_importance_A = pd.DataFrame({
    'feature': features_A,
    'importance': rf_model_A.feature_importances_
}).sort_values('importance', ascending=False)

print("\n상병코드 기반 모델 특징 중요도 (상위 10개):")
print(feature_importance_A.head(10))

# 진료과 기반 모델 특징 중요도
feature_importance_B = pd.DataFrame({
    'feature': features_B,
    'importance': rf_model_B.feature_importances_
}).sort_values('importance', ascending=False)

print("\n진료과 기반 모델 특징 중요도 (상위 10개):")
print(feature_importance_B.head(10))

# 특징 중요도 시각화
fig, axes = plt.subplots(1, 2, figsize=(15, 8))

# 상병코드 기반 모델
top_features_A = feature_importance_A.head(10)
sns.barplot(data=top_features_A, x='importance', y='feature', ax=axes[0], color='skyblue')
axes[0].set_title('상병코드 기반 모델 특징 중요도')

# 진료과 기반 모델
top_features_B = feature_importance_B.head(10)
sns.barplot(data=top_features_B, x='importance', y='feature', ax=axes[1], color='lightgreen')
axes[1].set_title('진료과 기반 모델 특징 중요도')

plt.tight_layout()
plt.savefig('new_merged_data/특징_중요도.png', dpi=300, bbox_inches='tight')
plt.show()

# ----------------------------------------------------------------------
# 7) 구간별 예측 성능 분석
# ----------------------------------------------------------------------
print("\n=== 구간별 예측 성능 분석 ===")

# 상병코드 기반 모델 구간별 성능
bin_accuracy_A = []
for bin_num in sorted(y_A.unique()):
    mask = y_test_A == bin_num
    if mask.sum() > 0:
        bin_acc = accuracy_score(y_test_A[mask], y_pred_A[mask])
        bin_accuracy_A.append(bin_acc)
        print(f"구간 {bin_num}: {bin_acc:.4f} (테스트 데이터 {mask.sum()}개)")

# 진료과 기반 모델 구간별 성능
bin_accuracy_B = []
for bin_num in sorted(y_B.unique()):
    mask = y_test_B == bin_num
    if mask.sum() > 0:
        bin_acc = accuracy_score(y_test_B[mask], y_pred_B[mask])
        bin_accuracy_B.append(bin_acc)
        print(f"구간 {bin_num}: {bin_acc:.4f} (테스트 데이터 {mask.sum()}개)")

# ----------------------------------------------------------------------
# 8) 결과 저장
# ----------------------------------------------------------------------
print("\n=== 결과 저장 ===")

# 모델 성능 결과 저장
performance_comparison.to_csv('new_merged_data/모델_성능_비교.csv', index=False, encoding='utf-8-sig')

# 특징 중요도 저장
feature_importance_A.to_csv('new_merged_data/상병코드_기반_특징중요도.csv', index=False, encoding='utf-8-sig')
feature_importance_B.to_csv('new_merged_data/진료과_기반_특징중요도.csv', index=False, encoding='utf-8-sig')

# 구간별 요약 정보 저장
bin_summary = df_with_cost.groupby('cost_bin').agg({
    '진료비(천원)': ['count', 'min', 'max', 'mean', 'median'],
    '상병코드': 'nunique',
    '진료과': 'nunique'
}).round(2)
bin_summary.columns = ['count', 'min_cost', 'max_cost', 'mean_cost', 'median_cost', 'unique_diseases', 'unique_depts']
bin_summary.to_csv('new_merged_data/구간별_상세_요약.csv', encoding='utf-8-sig')

print("모든 결과가 저장되었습니다!")

# ----------------------------------------------------------------------
# 9) 최종 요약
# ----------------------------------------------------------------------
print("\n=== 최종 요약 ===")
print(f"데이터 크기: {len(df_with_cost)}개")
print(f"구간 수: {df_with_cost['cost_bin'].nunique()}개")
print(f"상병코드 수: {df_with_cost['상병코드'].nunique()}개")
print(f"진료과 수: {df_with_cost['진료과'].nunique()}개")

print(f"\n상병코드 기반 모델:")
print(f"  - 정확도: {accuracy_A:.4f}")
print(f"  - 교차검증: {cv_scores_A.mean():.4f} (+/- {cv_scores_A.std() * 2:.4f})")

print(f"\n진료과 기반 모델:")
print(f"  - 정확도: {accuracy_B:.4f}")
print(f"  - 교차검증: {cv_scores_B.mean():.4f} (+/- {cv_scores_B.std() * 2:.4f})")

best_model = "상병코드 기반" if accuracy_A > accuracy_B else "진료과 기반"
print(f"\n더 나은 모델: {best_model}")

print("\n=== 작업 완료 ===") 