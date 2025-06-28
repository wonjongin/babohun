
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import scikit_posthocs as sp
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report


#자료 전처리
ekqlseh=pd.read_csv("C:/Users/jenny/babohun/new_merged_data/다빈도 질환 환자 연령별 분포_순위추가_합계계산_값통일.csv",encoding="utf-8-sig")
ekqlseh.loc[ekqlseh['구분'].str.contains('외래'), '연인원'] = \
    ekqlseh.loc[ekqlseh['구분'].str.contains('외래'), '실인원']
mask = ekqlseh['구분'] == '입원(실인원)'
rows_to_drop = ekqlseh[ekqlseh['구분'] == '입원(실인원)'].index
df_dropped = ekqlseh.drop(index=rows_to_drop)
cols_to_drop = ['순위', '상병명', '실인원']
df_result = df_dropped.drop(columns=cols_to_drop)
df_result
df_result.to_csv(
    "C:/Users/jenny/babohun/new_merged_data/df_result2.csv",
    index=False,
    encoding="utf-8-sig")
exclude_regions = ['서울', '대전', '대구']
df_filtered = df_result[~df_result['지역'].isin(exclude_regions)]


#아노바를 위한 검정 두개
# 1) 정규성 검정: 상병코드별로 Shapiro–Wilk 검정
print("=== 정규성 검정 (Shapiro–Wilk) ===")
for code, grp in df_filtered.groupby('상병코드'):
    values = grp['진료비(천원)'].dropna().values
    n = len(values)
    # 표본이 3 미만이면 건너뜀
    if n < 3:
        print(f"{code}: 표본수 {n} (<3), 검정 불가 → 건너뜀")
        continue
    # 표본이 500개 초과면 numpy로 샘플링
    if n > 500:
        values = np.random.choice(values, 500, replace=False)
    stat, p = stats.shapiro(values)
    print(f"{code}: W={stat:.4f}, p-value={p:.4e} (n={n})")

# 2) 등분산성 검정: Levene 테스트 (중앙값 기준)
groups = [grp['진료비(천원)'].dropna().values 
          for _, grp in df_filtered.groupby('상병코드') 
          if len(grp)>=2]
stat, p = stats.levene(*groups, center='median')
print(f"Levene H={stat:.4f}, p-value={p:.4e}")

'''
등분산 만족
정규성은 만족하지 않는 코드가 많음
->로그변환해보기
'''
# 1) 로그 변환 (0값 방지를 위해 +1)
df_filtered['log_cost'] = np.log(df_filtered['진료비(천원)'] + 1)
# 2) 변환 후 분포 확인 (예: 히스토그램, Q–Q 플롯)
plt.figure(figsize=(6,4))
plt.hist(df_filtered['log_cost'].dropna(), bins=30)
plt.title('Histogram of log_cost')
plt.xlabel('log(진료비(천원) + 1)')
plt.ylabel('Frequency')
plt.show()
# Q–Q 플롯 예시 (특정 상병코드 하나)
code = list(df_filtered['상병코드'].unique())[0]
values = df_filtered[df_filtered['상병코드']==code]['log_cost'].dropna().values
stats.probplot(values, dist="norm", plot=plt)
plt.title(f"Q–Q plot for {code}")
plt.show()
# 3) 로그 변환 후 Shapiro–Wilk 검
print("=== 정규성 검정 on log_cost (Shapiro–Wilk) ===")
for code, grp in df_filtered.groupby('상병코드'):
    values = grp['log_cost'].dropna().values
    n = len(values)
    if n < 3:
        print(f"{code}: n={n} (<3) → 검정 생략")
        continue
    if n > 500:
        values = np.random.choice(values, 500, replace=False)
    stat, p = stats.shapiro(values)
    print(f"{code}: W={stat:.4f}, p-value={p:.4e} (n={n})")

'''정규성 검정 실패
->비모수 방법으로 '''
# 1) Kruskal–Wallis 검정 (전체 상병코드 간 차이)
#    - 표본 수 3개 미만인 그룹은 제외
groups = [
    grp['진료비(천원)'].dropna().values
    for _, grp in df_filtered.groupby('상병코드')
    if len(grp['진료비(천원)'].dropna()) >= 3
]
stat, p = stats.kruskal(*groups)
print(f"Kruskal–Wallis H = {stat:.4f}, p-value = {p:.4e}")
if p < 0.05:
    print("→ 귀무가설 기각: 적어도 한 쌍의 상병코드 간 중앙값 차이가 유의합니다.\n")
else:
    print("→ 귀무가설 채택: 중앙값에 유의한 차이가 없습니다.\n")

# 2) Dunn’s 사후검정 (Bonferroni 보정)
#    - 전체 유의가 확인되면, 각 그룹 간 pairwise 비교
posthoc = sp.posthoc_dunn(
    df_filtered, 
    val_col='진료비(천원)', 
    group_col='상병코드', 
    p_adjust='bonferroni'
)
# p-value 행렬 출력 (DataFrame 형태)
print("=== Dunn’s post-hoc (Bonf.) p-value matrix ===")
print(posthoc)


#모델링(상위진료비 유발 질환코드 선정)
# 1) 레이블 생성
threshold = df_filtered['진료비(천원)'].quantile(0.75)
df_filtered['high_cost'] = (df_filtered['진료비(천원)'] >= threshold).astype(int)
# 2) 피처 준비 (상병코드 원-핫)
X = pd.get_dummies(df_filtered[['상병코드']], prefix='', prefix_sep='')
y = df_filtered['high_cost']
# 3) 학습/검증 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
# 4) 모델 학습
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)
# 5) 중요도 확인
importances = pd.Series(dt.feature_importances_, index=X.columns)
print("Top 10 중요 상병코드:\n", importances.nlargest(10))
# 6) 평가
from sklearn.metrics import classification_report
y_pred = dt.predict(X_test)
print(classification_report(y_test, y_pred))


#앙상블
# 1) 레이블 생성 (상위 25% 고비용)
threshold = df_filtered['진료비(천원)'].quantile(0.75)
df_filtered = df_filtered.copy()
df_filtered['high_cost'] = (df_filtered['진료비(천원)'] >= threshold).astype(int)
# 2) 피처 준비 (상병코드 원-핫 + 예시로 연령대·지역 포함)
X = pd.get_dummies(df_filtered[['상병코드', '지역']], dtype=int)
# 만약 연령대 컬럼(예: 'age_group')가 있다면:
# X = pd.get_dummies(df_filtered[['상병코드','지역','age_group']], dtype=int)
y = df_filtered['high_cost']
# 3) 학습/검증 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
# 4-1) Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("=== RandomForestClassifier ===")
print(classification_report(y_test, y_pred_rf))
# 4-2) Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
print("=== GradientBoostingClassifier ===")
print(classification_report(y_test, y_pred_gb))
# 5) 중요도 비교 (RF, GB)
importances_rf = pd.Series(rf.feature_importances_, index=X.columns)
importances_gb = pd.Series(gb.feature_importances_, index=X.columns)
print("RF 중요 상위 10:\n", importances_rf.nlargest(10))
print("\nGB 중요 상위 10:\n", importances_gb.nlargest(10))
# top10 상병코드만
importances_rf = pd.Series(rf.feature_importances_, index=X.columns)
importances_gb = pd.Series(gb.feature_importances_, index=X.columns)
print("RF: 상병코드 중요도 Top 10")
print(importances_rf[importances_rf.index.str.startswith('code_')]
      .nlargest(10))
print("\nGB: 상병코드 중요도 Top 10")
print(importances_gb[importances_gb.index.str.startswith('code_')]
      .nlargest(10))

'''
1. **데이터 전처리**  
   - df_result에서 서울·대전·대구 제외 → df_filtered  
   - 사용 컬럼: 상병코드, 지역, 진료비(천원)

2. **ANOVA 전제조건 검정**  
   - Shapiro–Wilk: 다수 상병코드에서 p < 0.05 → 정규성 위배  
   - Levene: p ≫ 0.05 → 등분산성 만족  
   → **정규성 위배**로 비모수 검정 선택

3. **비모수 검정**  
   - Kruskal–Wallis: H=73.19, p=0.0066 → 그룹 간 중앙값 차이 존재  
   - Dunn’s (Bonferroni): 모든 p=1.0 → 다중비교 보수성·샘플 수 한계로 개별 차이 불검출

4. **단일 모델(의사결정나무)**  
   - 레이블: 진료비 상위 25% → high_cost  
   - 피처: 상병코드 원-핫  
   - **Top-4 중요코드**: M48, N18, Z49, F00  
   - 성능: Accuracy=0.76, 고비용 Recall=0.07

5. **클래스 불균형 대응**  
   - `class_weight='balanced'`·RandomOverSampler → 성능 변화 거의 없음

6. **앙상블 모델**  
   - 피처: 상병코드(`code_…`) + 지역(`region_…`)  
   - **RandomForest**  
     - Accuracy=0.87, 고비용 Recall=0.96, F1=0.79  
     - 중요 상위 코드: M48, N18, Z49, M17, I63  
   - **GradientBoosting**  
     - Accuracy=0.89, 고비용 Recall=0.93, F1=0.81  
     - 중요 상위 코드: M48, N18, Z49, M17, I63  

7. **최종 “상위진료비 유발 질환코드” 선정**  
   - **가장 일관되게 중요도가 높았던 5개 코드**  
     1. **M48** (척추관협착 등)  
     2. **N18** (만성 신부전)  
     3. **Z49** (투석 관련)  
     4. **M17** (골관절염)  
     5. **I63** (뇌경색)  

---

### 이 결과가 목표에 부합하는가?  
네—  
“상위진료비 유발 질환코드 선정”이라는 목적에 맞춰  
- 비모수 검정으로 그룹 간 차이를 확인하고,  
- 단일·앙상블 모델을 통해 실제 ‘고비용 여부’ 예측에서  
  **반복적으로 중요하게 나온 코드**를 최종 선별했습니다.  

다음 단계로는  
- 하이퍼파라미터 튜닝  
- SHAP 분석 등으로 개별 환자 예측 기여도 해석  
을 진행하시면 더욱 탄탄한 결과를 얻으실 수 있습니다.
'''