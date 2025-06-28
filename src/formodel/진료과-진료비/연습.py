import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report


# 1) 원본 데이터 불러오기
ekqlseh = pd.read_csv(
    "C:/Users/jenny/babohun/new_merged_data/다빈도 질환 환자 연령별 분포_순위추가_합계계산_값통일.csv",
    encoding="utf-8-sig"
)

# 2) 외래인원 컬럼 병합, 입원(실인원) 행 제거
ekqlseh.loc[ekqlseh['구분'].str.contains('외래'), '연인원'] = ekqlseh['실인원']
ekqlseh = ekqlseh[~(ekqlseh['구분'] == '입원(실인원)')]

# 3) 불필요한 컬럼 제거
df_result = ekqlseh.drop(columns=['순위', '상병명', '실인원'])

# 4) 지역 필터링
exclude_regions = ['서울', '대전', '대구']
df_filtered = df_result[~df_result['지역'].isin(exclude_regions)].copy()

# 5) 컬럼명 변경: '진료비(천원)' → '진료비'
df_filtered.rename(columns={'진료비(천원)': '진료비'}, inplace=True)

# 6) 상병코드 ↔ 진료과 매핑 테이블 불러오기 및 병합
mapping = pd.read_csv(
    "C:/Users/jenny/babohun/df_result2_with_심평원.csv",
    encoding="utf-8-sig"
)
df_filtered = (
    df_filtered
    .merge(mapping[['상병코드', '진료과']], on='상병코드', how='left')
    .dropna(subset=['진료과'])  # 매핑 안 된 행 제거
)


# 3) 고비용 여부 레이블 생성 (상위 25%)
threshold = df['진료비'].quantile(0.75)
df['high_cost'] = (df['진료비'] >= threshold).astype(int)

# 4) 피처 준비: 진료과 원-핫만 사용
X = pd.get_dummies(df['진료과'], prefix='dept')
y = df['high_cost']

# 5) 학습/검증 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)

# 6) 모델 학습 & 중요도 추출 함수
def fit_and_report(model, name):
    model.fit(X_train, y_train)
    print(f"\n=== {name} ===")
    print(classification_report(y_test, model.predict(X_test)))
    imps = pd.Series(model.feature_importances_, index=X.columns)
    top = imps.nlargest(10)
    print(f"{name} 중요 진료과 Top 10:\n{top}\n")
    return top

# 7) Decision Tree
dt = DecisionTreeClassifier(max_depth=4, class_weight='balanced', random_state=42)
top_dt = fit_and_report(dt, "Decision Tree")

# 8) Random Forest
rf = RandomForestClassifier(
    n_estimators=200, max_depth=6,
    class_weight='balanced', random_state=42, n_jobs=-1
)
top_rf = fit_and_report(rf, "Random Forest")

# 9) Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42
)
top_gb = fit_and_report(gb, "Gradient Boosting")

# 10) 최종 “상위진료비 유발 진료과” 선정
# 세 모델에서 공통으로 중요도가 높았던 진료과를 뽑아봅니다.
common = set(top_dt.index) & set(top_rf.index) & set(top_gb.index)
print("🚀 최종 상위진료비 유발 진료과 (공통 Top 중요도):", common)
