# -*- coding: utf-8 -*-
"""
진료비 구간 분류 모델 v2
5개 구간 + 다양한 모델 비교
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Pretendard'
plt.rcParams['axes.unicode_minus'] = False

# ----------------------------------------------------------------------
# 1) 데이터 로드 및 전처리
# ----------------------------------------------------------------------
print("=== 데이터 로드 및 전처리 ===")

# 원본 데이터 로드
df = pd.read_csv("new_merged_data/df_result2_with_심평원_진료비.csv", encoding="utf-8-sig")

print(f"전체 데이터: {len(df)}개")
print(f"컬럼: {list(df.columns)}")

# 진료비가 있는 데이터만 사용
df_with_cost = df[df['진료비(천원)'].notna()].copy()
print(f"진료비가 있는 데이터: {len(df_with_cost)}개")

# ----------------------------------------------------------------------
# 2) 5개 구간으로 재분류
# ----------------------------------------------------------------------
print("\n=== 5개 구간으로 재분류 ===")

# 진료비 분포 확인
print("진료비 통계:")
print(df_with_cost['진료비(천원)'].describe())

# 5개 구간으로 분류 (로그 스케일 기반)
log_costs = np.log10(df_with_cost['진료비(천원)'])
bins = np.linspace(log_costs.min(), log_costs.max(), 6)
labels = [0, 1, 2, 3, 4]

df_with_cost['cost_bin_5'] = pd.cut(log_costs, bins=bins, labels=labels, include_lowest=True)
df_with_cost['cost_bin_5'] = df_with_cost['cost_bin_5'].astype(int)

# 구간별 통계
bin_stats = df_with_cost.groupby('cost_bin_5')['진료비(천원)'].agg(['count', 'min', 'max', 'mean', 'median']).round(2)
print("\n5개 구간별 통계:")
print(bin_stats)

# 구간별 분포 확인
print(f"\n구간별 분포:")
print(df_with_cost['cost_bin_5'].value_counts().sort_index())

# ----------------------------------------------------------------------
# 3) 특징 엔지니어링
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

# 상병코드 그룹화 (ICD-10 대분류 기준)
def get_disease_category(disease_code):
    if pd.isna(disease_code):
        return 'Unknown'
    code = str(disease_code)
    if code.startswith('A') or code.startswith('B'):
        return 'Infectious'
    elif code.startswith('C'):
        return 'Neoplasm'
    elif code.startswith('D'):
        return 'Blood'
    elif code.startswith('E'):
        return 'Endocrine'
    elif code.startswith('F'):
        return 'Mental'
    elif code.startswith('G'):
        return 'Nervous'
    elif code.startswith('H'):
        return 'Eye_Ear'
    elif code.startswith('I'):
        return 'Circulatory'
    elif code.startswith('J'):
        return 'Respiratory'
    elif code.startswith('K'):
        return 'Digestive'
    elif code.startswith('L'):
        return 'Skin'
    elif code.startswith('M'):
        return 'Musculoskeletal'
    elif code.startswith('N'):
        return 'Genitourinary'
    elif code.startswith('O'):
        return 'Pregnancy'
    elif code.startswith('P'):
        return 'Perinatal'
    elif code.startswith('Q'):
        return 'Congenital'
    elif code.startswith('R'):
        return 'Symptoms'
    elif code.startswith('S') or code.startswith('T'):
        return 'Injury'
    elif code.startswith('Z'):
        return 'Health_Status'
    else:
        return 'Other'

df_with_cost['disease_category'] = df_with_cost['상병코드'].apply(get_disease_category)

print("특징 엔지니어링 완료")

# ----------------------------------------------------------------------
# 4) 모델 A: 상병코드 기반 진료비 구간 예측
# ----------------------------------------------------------------------
print("\n=== 모델 A: 상병코드 기반 진료비 구간 예측 ===")

# 상병코드 인코딩
le_disease = LabelEncoder()
df_with_cost['disease_encoded'] = le_disease.fit_transform(df_with_cost['상병코드'])

# 질병 카테고리 인코딩
le_category = LabelEncoder()
df_with_cost['category_encoded'] = le_category.fit_transform(df_with_cost['disease_category'])

# 특징 선택
features_A = [
    'disease_encoded', 'category_encoded', 'year_encoded', 'is_inpatient', '연인원',
    '59이하_ratio', '60-64_ratio', '65-69_ratio', '70-79_ratio', '80-89_ratio', '90이상_ratio',
    'age_diversity'
]

X_A = df_with_cost[features_A]
y_A = df_with_cost['cost_bin_5']

# NaN 값 처리
imputer = SimpleImputer(strategy='mean')
X_A_imputed = pd.DataFrame(imputer.fit_transform(X_A), columns=X_A.columns)

print(f"상병코드 기반 모델 특징: {len(features_A)}개")
print(f"목표 변수 분포:\n{y_A.value_counts().sort_index()}")

# 데이터 분할
X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(
    X_A_imputed, y_A, test_size=0.2, random_state=42, stratify=y_A
)

# 스케일링
scaler_A = StandardScaler()
X_train_A_scaled = scaler_A.fit_transform(X_train_A)
X_test_A_scaled = scaler_A.transform(X_test_A)

# ----------------------------------------------------------------------
# 5) 모델 B: 진료과 기반 진료비 구간 예측
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
y_B = df_with_cost['cost_bin_5']

# NaN 값 처리
X_B_imputed = pd.DataFrame(imputer.fit_transform(X_B), columns=X_B.columns)

print(f"진료과 기반 모델 특징: {len(features_B)}개")

# 데이터 분할
X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(
    X_B_imputed, y_B, test_size=0.2, random_state=42, stratify=y_B
)

# 스케일링
scaler_B = StandardScaler()
X_train_B_scaled = scaler_B.fit_transform(X_train_B)
X_test_B_scaled = scaler_B.transform(X_test_B)

# ----------------------------------------------------------------------
# 6) 다양한 모델 학습 및 비교
# ----------------------------------------------------------------------
print("\n=== 다양한 모델 학습 및 비교 ===")

# 모델 정의 (SVM 제거하고 다른 모델들 사용)
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

# 결과 저장용 딕셔너리
results = {
    'Model': [],
    'Type': [],
    'Accuracy': [],
    'F1_Score': [],
    'CV_Accuracy': [],
    'CV_Std': []
}

# 상병코드 기반 모델들 학습
print("\n상병코드 기반 모델 학습 중...")
for name, model in models.items():
    print(f"  - {name} 학습 중...")
    
    # 스케일링이 필요한 모델들
    if name in ['Neural Network', 'Logistic Regression']:
        model.fit(X_train_A_scaled, y_train_A)
        y_pred = model.predict(X_test_A_scaled)
        cv_scores = cross_val_score(model, X_A_imputed, y_A, cv=5, scoring='accuracy')
    else:
        model.fit(X_train_A, y_train_A)
        y_pred = model.predict(X_test_A)
        cv_scores = cross_val_score(model, X_A_imputed, y_A, cv=5, scoring='accuracy')
    
    accuracy = accuracy_score(y_test_A, y_pred)
    f1 = f1_score(y_test_A, y_pred, average='weighted')
    
    results['Model'].append(name)
    results['Type'].append('상병코드 기반')
    results['Accuracy'].append(accuracy)
    results['F1_Score'].append(f1)
    results['CV_Accuracy'].append(cv_scores.mean())
    results['CV_Std'].append(cv_scores.std())

# 진료과 기반 모델들 학습
print("\n진료과 기반 모델 학습 중...")
for name, model in models.items():
    print(f"  - {name} 학습 중...")
    
    # 스케일링이 필요한 모델들
    if name in ['Neural Network', 'Logistic Regression']:
        model.fit(X_train_B_scaled, y_train_B)
        y_pred = model.predict(X_test_B_scaled)
        cv_scores = cross_val_score(model, X_B_imputed, y_B, cv=5, scoring='accuracy')
    else:
        model.fit(X_train_B, y_train_B)
        y_pred = model.predict(X_test_B)
        cv_scores = cross_val_score(model, X_B_imputed, y_B, cv=5, scoring='accuracy')
    
    accuracy = accuracy_score(y_test_B, y_pred)
    f1 = f1_score(y_test_B, y_pred, average='weighted')
    
    results['Model'].append(name)
    results['Type'].append('진료과 기반')
    results['Accuracy'].append(accuracy)
    results['F1_Score'].append(f1)
    results['CV_Accuracy'].append(cv_scores.mean())
    results['CV_Std'].append(cv_scores.std())

# 결과를 DataFrame으로 변환
results_df = pd.DataFrame(results)
results_df = results_df.round(4)

print("\n=== 모델 성능 비교 결과 ===")
print(results_df)

# ----------------------------------------------------------------------
# 7) 최고 성능 모델 상세 분석
# ----------------------------------------------------------------------
print("\n=== 최고 성능 모델 상세 분석 ===")

# 최고 정확도 모델 찾기
best_model_idx = results_df['Accuracy'].idxmax()
best_model_name = results_df.loc[best_model_idx, 'Model']
best_model_type = results_df.loc[best_model_idx, 'Type']

print(f"최고 성능 모델: {best_model_name} ({best_model_type})")
print(f"정확도: {results_df.loc[best_model_idx, 'Accuracy']:.4f}")
print(f"F1 점수: {results_df.loc[best_model_idx, 'F1_Score']:.4f}")

# 최고 성능 모델 재학습 및 상세 분석
if best_model_type == '상병코드 기반':
    X_train, X_test, y_train, y_test = X_train_A, X_test_A, y_train_A, y_test_A
    X_train_scaled, X_test_scaled = X_train_A_scaled, X_test_A_scaled
    features = features_A
else:
    X_train, X_test, y_train, y_test = X_train_B, X_test_B, y_train_B, y_test_B
    X_train_scaled, X_test_scaled = X_train_B_scaled, X_test_B_scaled
    features = features_B

# 최고 성능 모델 재학습
best_model = models[best_model_name]
if best_model_name in ['Neural Network', 'Logistic Regression']:
    best_model.fit(X_train_scaled, y_train)
    y_pred = best_model.predict(X_test_scaled)
else:
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

# 분류 보고서
print(f"\n{best_model_name} ({best_model_type}) 분류 보고서:")
print(classification_report(y_test, y_pred))

# 구간별 성능
print(f"\n{best_model_name} ({best_model_type}) 구간별 성능:")
for bin_num in sorted(y_test.unique()):
    mask = y_test == bin_num
    if mask.sum() > 0:
        bin_acc = accuracy_score(y_test[mask], y_pred[mask])
        print(f"구간 {bin_num}: {bin_acc:.4f} (테스트 데이터 {mask.sum()}개)")

# ----------------------------------------------------------------------
# 8) 시각화
# ----------------------------------------------------------------------
print("\n=== 시각화 ===")

# 1. 모델 성능 비교 차트
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 정확도 비교
accuracy_pivot = results_df.pivot(index='Model', columns='Type', values='Accuracy')
accuracy_pivot.plot(kind='bar', ax=axes[0,0], color=['skyblue', 'lightgreen'])
axes[0,0].set_title('모델별 정확도 비교')
axes[0,0].set_ylabel('정확도')
axes[0,0].legend()
axes[0,0].tick_params(axis='x', rotation=45)

# F1 점수 비교
f1_pivot = results_df.pivot(index='Model', columns='Type', values='F1_Score')
f1_pivot.plot(kind='bar', ax=axes[0,1], color=['skyblue', 'lightgreen'])
axes[0,1].set_title('모델별 F1 점수 비교')
axes[0,1].set_ylabel('F1 점수')
axes[0,1].legend()
axes[0,1].tick_params(axis='x', rotation=45)

# 교차 검증 정확도 비교
cv_pivot = results_df.pivot(index='Model', columns='Type', values='CV_Accuracy')
cv_pivot.plot(kind='bar', ax=axes[1,0], color=['skyblue', 'lightgreen'])
axes[1,0].set_title('모델별 교차 검증 정확도 비교')
axes[1,0].set_ylabel('교차 검증 정확도')
axes[1,0].legend()
axes[1,0].tick_params(axis='x', rotation=45)

# 상병코드 vs 진료과 기반 비교
type_comparison = results_df.groupby('Type')['Accuracy'].mean()
type_comparison.plot(kind='bar', ax=axes[1,1], color=['skyblue', 'lightgreen'])
axes[1,1].set_title('상병코드 vs 진료과 기반 평균 정확도')
axes[1,1].set_ylabel('평균 정확도')
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('new_merged_data/모델_성능_비교_v2.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. 최고 성능 모델 혼동 행렬
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'{best_model_name} ({best_model_type}) 혼동 행렬')
plt.xlabel('예측 구간')
plt.ylabel('실제 구간')
plt.savefig('new_merged_data/최고성능_모델_혼동행렬.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. 구간별 분포
plt.figure(figsize=(10, 6))
df_with_cost['cost_bin_5'].value_counts().sort_index().plot(kind='bar', color='lightcoral')
plt.title('5개 구간별 데이터 분포')
plt.xlabel('진료비 구간')
plt.ylabel('데이터 개수')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('new_merged_data/5구간_분포.png', dpi=300, bbox_inches='tight')
plt.show()

# ----------------------------------------------------------------------
# 9) 결과 저장
# ----------------------------------------------------------------------
print("\n=== 결과 저장 ===")

# 모델 성능 결과 저장
results_df.to_csv('new_merged_data/모델_성능_비교_v2.csv', index=False, encoding='utf-8-sig')

# 5구간 데이터 저장
df_with_cost.to_csv('new_merged_data/df_result2_with_심평원_진료비_5구간.csv', index=False, encoding='utf-8-sig')

# 구간별 요약 정보 저장
bin_summary_5 = df_with_cost.groupby('cost_bin_5').agg({
    '진료비(천원)': ['count', 'min', 'max', 'mean', 'median'],
    '상병코드': 'nunique',
    '진료과': 'nunique',
    'disease_category': 'nunique'
}).round(2)
bin_summary_5.columns = ['count', 'min_cost', 'max_cost', 'mean_cost', 'median_cost', 'unique_diseases', 'unique_depts', 'unique_categories']
bin_summary_5.to_csv('new_merged_data/5구간별_상세_요약.csv', encoding='utf-8-sig')

print("모든 결과가 저장되었습니다!")

# ----------------------------------------------------------------------
# 10) 최종 요약
# ----------------------------------------------------------------------
print("\n=== 최종 요약 ===")
print(f"데이터 크기: {len(df_with_cost)}개")
print(f"구간 수: {df_with_cost['cost_bin_5'].nunique()}개")
print(f"상병코드 수: {df_with_cost['상병코드'].nunique()}개")
print(f"진료과 수: {df_with_cost['진료과'].nunique()}개")
print(f"질병 카테고리 수: {df_with_cost['disease_category'].nunique()}개")

print(f"\n최고 성능 모델:")
print(f"  - 모델: {best_model_name}")
print(f"  - 타입: {best_model_type}")
print(f"  - 정확도: {results_df.loc[best_model_idx, 'Accuracy']:.4f}")
print(f"  - F1 점수: {results_df.loc[best_model_idx, 'F1_Score']:.4f}")
print(f"  - 교차검증: {results_df.loc[best_model_idx, 'CV_Accuracy']:.4f} (+/- {results_df.loc[best_model_idx, 'CV_Std'] * 2:.4f})")

print(f"\n모델별 평균 성능:")
print(f"  - 상병코드 기반: {results_df[results_df['Type']=='상병코드 기반']['Accuracy'].mean():.4f}")
print(f"  - 진료과 기반: {results_df[results_df['Type']=='진료과 기반']['Accuracy'].mean():.4f}")

print("\n=== 작업 완료 ===") 