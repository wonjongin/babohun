import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import scikit_posthocs as sp
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

# --- ANOVA 전제조건 검정 (진료과별) ---
print("=== 정규성 검정 (Shapiro–Wilk) by 진료과 ===")
for dept, grp in df_filtered.groupby('진료과'):
    vals = grp['진료비'].dropna().values
    n = len(vals)
    if n < 3:
        print(f"{dept}: n={n} (<3), 검정 생략")
        continue
    if n > 500:
        vals = np.random.choice(vals, 500, replace=False)
    stat, p = stats.shapiro(vals)
    print(f"{dept}: W={stat:.4f}, p-value={p:.4e} (n={n})")

# 등분산성 검정 (Levene)
groups = [
    grp['진료비'].dropna().values
    for _, grp in df_filtered.groupby('진료과')
    if len(grp) >= 2
]
stat, p = stats.levene(*groups, center='median')
print(f"Levene H={stat:.4f}, p-value={p:.4e}")

# --- 로그 변환 후 재검정 ---
df_filtered['log_cost'] = np.log(df_filtered['진료비'] + 1)

plt.figure(figsize=(6,4))
plt.hist(df_filtered['log_cost'], bins=30)
plt.title('Histogram of log_cost')
plt.xlabel('log(진료비 + 1)')
plt.ylabel('Frequency')
plt.show()

print("=== 정규성 검정 on log_cost by 진료과 ===")
for dept, grp in df_filtered.groupby('진료과'):
    vals = grp['log_cost'].dropna().values
    n = len(vals)
    if n < 3:
        print(f"{dept}: n={n} (<3), 생략")
        continue
    if n > 500:
        vals = np.random.choice(vals, 500, replace=False)
    stat, p = stats.shapiro(vals)
    print(f"{dept}: W={stat:.4f}, p-value={p:.4e} (n={n})")
# --- 로그 변환 (이미 수행된 상태라고 가정) ---
# df_filtered['log_cost'] = np.log(df_filtered['진료비'] + 1)

# 1) Fligner–Killeen 비모수 등분산성 검정 (log_cost 기준)
fl_stat, fl_p = stats.fligner(
    *[grp['log_cost'].dropna().values 
      for _, grp in df_filtered.groupby('진료과') 
      if len(grp['log_cost'].dropna()) >= 2]
)
print(f"Fligner–Killeen H={fl_stat:.4f}, p-value={fl_p:.4e}")

if fl_p >= 0.05:
    # 2-A) 등분산성 만족 → Kruskal–Wallis & Dunn’s
    print("→ 등분산성 만족: Kruskal–Wallis 검정 수행")
    groups = [
        grp['log_cost'].dropna().values
        for _, grp in df_filtered.groupby('진료과')
        if len(grp['log_cost'].dropna()) >= 3
    ]
    kw_stat, kw_p = stats.kruskal(*groups)
    print(f"Kruskal–Wallis H={kw_stat:.4f}, p-value={kw_p:.4e}")
    if kw_p < 0.05:
        print("→ 귀무가설 기각: 적어도 한 진료과 간 중앙값 차이 유의")
    else:
        print("→ 귀무가설 채택: 중앙값 차이 없음")
    
    # Dunn’s 사후검정
    posthoc = sp.posthoc_dunn(
        df_filtered,
        val_col='log_cost',
        group_col='진료과',
        p_adjust='bonferroni'
    )
    print("=== Dunn’s post-hoc (Bonferroni) p-value matrix ===")
    print(posthoc)

else:
    # 2-B) 등분산성 위배 → Permutation ANOVA (비모수적 대안)
    print("→ 등분산성 위배: Permutation ANOVA 수행")
    def perm_anova(data, labels, n_perm=5000):
        # 관측된 F-통계량
        obs_F = stats.f_oneway(
            *[data[labels == g] for g in np.unique(labels)]
        ).statistic
        count = 0
        for _ in range(n_perm):
            perm_labels = np.random.permutation(labels)
            F = stats.f_oneway(
                *[data[perm_labels == g] for g in np.unique(labels)]
            ).statistic
            if F >= obs_F:
                count += 1
        return obs_F, count / n_perm

    costs = df_filtered['log_cost'].values
    labs  = df_filtered['진료과'].values
    perm_F, perm_p = perm_anova(costs, labs, n_perm=10000)
    print(f"Permutation ANOVA F={perm_F:.4f}, p-value={perm_p:.4f}")

# --- 비모수 검정 (Kruskal–Wallis & Dunn's) ---
groups = [
    grp['진료비'].dropna().values
    for _, grp in df_filtered.groupby('진료과')
    if len(grp['진료비'].dropna()) >= 3
]
stat, p = stats.kruskal(*groups)
print(f"Kruskal–Wallis H={stat:.4f}, p-value={p:.4e}")
if p < 0.05:
    print("→ 귀무가설 기각: 적어도 한 진료과 간 중앙값 차이 유의")
else:
    print("→ 귀무가설 채택: 중앙값 차이 없음")

# Dunn’s post-hoc
posthoc = sp.posthoc_dunn(
    df_filtered,
    val_col='진료비',
    group_col='진료과',
    p_adjust='bonferroni'
)
print("=== Dunn’s post-hoc (Bonferroni) p-value matrix by 진료과 ===")
print(posthoc)

# --- 모델링 (진료과 기반) ---
# 1) 레이블
threshold = df_filtered['진료비'].quantile(0.75)
df_filtered['high_cost'] = (df_filtered['진료비'] >= threshold).astype(int)

# 2) 피처: 진료과 원-핫
X = pd.get_dummies(df_filtered[['진료과', '지역']], dtype=int)
y = df_filtered['high_cost']

# 3) 학습/검증 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 4-1) Decision Tree
dt = DecisionTreeClassifier(max_depth=4, class_weight='balanced', random_state=42)
dt.fit(X_train, y_train)
print("Decision Tree 중요 진료과:\n", 
      pd.Series(dt.feature_importances_, index=X.columns).nlargest(10))
y_pred = dt.predict(X_test)
print(classification_report(y_test, y_pred))

# 4-2) Random Forest & Gradient Boosting
rf = RandomForestClassifier(
    n_estimators=200, max_depth=6, class_weight='balanced',
    random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
print("=== RandomForestClassifier ===")
print(classification_report(y_test, rf.predict(X_test)))

gb = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42
)
gb.fit(X_train, y_train)
print("=== GradientBoostingClassifier ===")
print(classification_report(y_test, gb.predict(X_test)))

# 중요도 비교
imp_rf = pd.Series(rf.feature_importances_, index=X.columns).nlargest(10)
imp_gb = pd.Series(gb.feature_importances_, index=X.columns).nlargest(10)
print("RF 중요 진료과 Top 10:\n", imp_rf)
print("GB 중요 진료과 Top 10:\n", imp_gb)
