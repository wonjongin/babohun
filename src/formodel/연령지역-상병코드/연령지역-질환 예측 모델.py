import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    top_k_accuracy_score, balanced_accuracy_score, roc_auc_score
)
from sklearn.preprocessing import label_binarize

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1) 데이터 적재 및 가공
# --------------------------------------------------
df = pd.read_csv("new_merged_data/df_result2_mapping1.csv", dtype=str)
age_cols = ["59이하", "60-64", "65-69", "70-79", "80-89", "90이상"]

m = df.melt(
    id_vars=["년도", "구분", "지역", "상병코드", "진료과"],
    value_vars=age_cols,
    var_name="age_group",
    value_name="count",
)
m["count"] = pd.to_numeric(m["count"], errors="coerce").fillna(0).astype(int)
m["대표진료과"] = m["진료과"]
train = m[m["대표진료과"].notna()]

train["year_num"] = train["년도"].astype(int) - train["년도"].astype(int).min()

X = train[["year_num", "지역", "age_group", "구분"]]
y = train["대표진료과"]
w = train["count"]

# --------------------------------------------------
# 2) 학습 / 검증 분리
# --------------------------------------------------
X_tr, X_te, y_tr, y_te, w_tr, w_te = train_test_split(
    X, y, w, test_size=0.20, stratify=y, random_state=42
)

# --------------------------------------------------
# 3) 전처리 파이프라인
# --------------------------------------------------
cat_cols = ["지역", "age_group", "구분"]
num_cols = ["year_num"]

preprocessor = ColumnTransformer(
    [
        ("ohe", OneHotEncoder(drop="first", sparse_output=False), cat_cols),
        ("scale", StandardScaler(), num_cols),
    ]
)

# --------------------------------------------------
# 4) XGB용 래퍼 클래스
# --------------------------------------------------
class XGBWrapper(XGBClassifier):
    """문자→숫자 라벨을 내부 변환하고, 원본 라벨은 orig_classes_에 저장"""
    def fit(self, X, y, **kwargs):
        self._le = LabelEncoder()
        y_enc = self._le.fit_transform(y)
        super().fit(X, y_enc, **kwargs)
        self.orig_classes_ = self._le.classes_   # <- 여기만 남깁니다
        return self

    def predict(self, X):
        return self._le.inverse_transform(super().predict(X))

    def predict_proba(self, X):
        return super().predict_proba(X)

# --------------------------------------------------
# 5) 파이프라인 & 그리드 정의 함수
# --------------------------------------------------
def make_pipeline(clf, param_grid):
    pipe = ImbPipeline(
        [
            ("prep", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("select", SelectKBest(f_classif)),
            ("clf", clf),
        ]
    )
    return pipe, param_grid


# Logistic Regression
pipe_lr, params_lr = make_pipeline(
    LogisticRegression(
        penalty="l1", solver="saga", max_iter=2000, class_weight="balanced"
    ),
    {
        "select__k": [30, 50, 100],
        "clf__C": [0.01, 0.1, 1],
    },
)

# Random Forest
pipe_rf, params_rf = make_pipeline(
    RandomForestClassifier(class_weight="balanced", random_state=42),
    {
        "select__k": [30, 50, 100],
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [None, 10, 20],
    },
)

# XGBoost – sample_weight 제외
pipe_xgb, params_xgb = make_pipeline(
    XGBWrapper(
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        tree_method="hist",   # CPU 학습 속도 향상
    ),
    {
        "select__k": [30, 50, 100],
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [3, 6],
        "clf__learning_rate": [0.01, 0.1],
        "clf__reg_alpha": [0, 1],
    },
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --------------------------------------------------
# 6) 그리드 서치 실행
# --------------------------------------------------
grids = {}
for name, (pipe, params) in zip(
    ["lr", "rf", "xgb"],
    [
        (pipe_lr, params_lr),
        (pipe_rf, params_rf),
        (pipe_xgb, params_xgb),
    ],
):
    grid = GridSearchCV(
        pipe, params, cv=cv, scoring="accuracy", n_jobs=-1, verbose=0
    )
    grid.fit(X_tr, y_tr)  # sample_weight 미사용
    grids[name] = grid

# --------------------------------------------------
# 7) 앙상블 (Voting & Stacking)
# --------------------------------------------------
estimators = [(n, grids[n].best_estimator_) for n in ["lr", "rf", "xgb"]]

vot = VotingClassifier(estimators=estimators, voting="soft")
vot.fit(X_tr, y_tr)

stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=cv,
    n_jobs=-1,
)
stack.fit(X_tr, y_tr)

# --------------------------------------------------
# 8) 평가 함수
# --------------------------------------------------
def eval_model(name, model, X, y_true, w):
    y_pred = model.predict(X)
    proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None

    print(f"\n=== {name} ===")
    print("Acc:", accuracy_score(y_true, y_pred, sample_weight=w))
    print("Macro-F1:", f1_score(y_true, y_pred, average="macro", sample_weight=w))

    if proba is not None:
        # ---- class_order 안전 추출 ----
        class_order = getattr(model, "orig_classes_", None)
        if class_order is None:
            class_order = getattr(model, "classes_", None)
        if class_order is None:
            class_order = np.unique(y_true)
        # --------------------------------

        print("Top-3 Acc:",
              top_k_accuracy_score(y_true, proba, k=3, sample_weight=w))
        print("Bal Acc:",
              balanced_accuracy_score(y_true, y_pred, sample_weight=w))

        y_bin = label_binarize(y_true, classes=class_order)
        print("Macro-ROC-AUC:",
              roc_auc_score(y_bin, proba, average="macro", sample_weight=w))

    print(classification_report(y_true, y_pred, sample_weight=w, digits=3))

for nm, mdl in [
    ("XGB", grids["xgb"].best_estimator_),
    ("Voting", vot),
    ("Stacking", stack),
]:
    eval_model(nm, mdl, X_te, y_te, w_te)


'''
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

# 1) 데이터 로드 & melt
df = pd.read_csv("new_merged_data/df_result2_mapping1.csv", dtype=str)
age_cols = ['59이하','60-64','65-69','70-79','80-89','90이상']
m = df.melt(
    id_vars=['년도','구분','지역','상병코드','진료과'],
    value_vars=age_cols,
    var_name='age_group',
    value_name='count'
)
m['count'] = (
    pd.to_numeric(m['count'], errors='coerce')
      .fillna(0)
      .astype(int)
)
m['대표진료과'] = m['진료과']
train = m[m['대표진료과'].notna()]

# 2) 주효과 더미 인코딩 (년도 포함, drop_first=True)
X0 = pd.get_dummies(
    train[['년도','지역','age_group','구분']],
    drop_first=True
).astype(float)

# 3) 선택적 interaction 항 생성
X_int = X0.copy()
# 컬럼 분류
year_cols         = [c for c in X0.columns if c.startswith('년도_')]
region_cols       = [c for c in X0.columns if c.startswith('지역_')]
age_group_cols    = [c for c in X0.columns if c.startswith('age_group_')]
distinction_cols  = [c for c in X0.columns if c.startswith('구분_')]

# 3a) 기존 교호작용 유지
for r in region_cols:
    for a in age_group_cols:
        X_int[f"{r}*{a}"] = X0[r] * X0[a]
    for d in distinction_cols:
        X_int[f"{r}*{d}"] = X0[r] * X0[d]

for a in age_group_cols:
    for d in distinction_cols:
        X_int[f"{a}*{d}"] = X0[a] * X0[d]

# 3b) 년도와의 교호작용 추가
for ycol in year_cols:
    # 년도×지역
    for r in region_cols:
        X_int[f"{ycol}*{r}"] = X0[ycol] * X0[r]
    # 년도×연령대
    for a in age_group_cols:
        X_int[f"{ycol}*{a}"] = X0[ycol] * X0[a]
    # 년도×구분
    for d in distinction_cols:
        X_int[f"{ycol}*{d}"] = X0[ycol] * X0[d]

# 4) 절편 추가 및 VIF 계산
X_const = sm.add_constant(X_int)
vif_full = pd.DataFrame({
    'feature': X_const.columns,
    'VIF': [
        variance_inflation_factor(X_const.values, i)
        for i in range(X_const.shape[1])
    ]
})
print(vif_full.to_string(index=False))

# 5) VIF > 10인 interaction 항만 추출(확인용)
high_vif = vif_full[
    vif_full['feature'].str.contains(r'\*') & (vif_full['VIF'] > 10)
]['feature'].tolist()
print("\n=== VIF > 10인 interaction 항 ===\n", high_vif)

# 6) 모델 학습 데이터 준비
X = X_int.copy()
y = train['대표진료과'].values
w = train['count'].values.astype(float)

# 7) train/test split
X_tr, X_te, y_tr, y_te, w_tr, w_te = train_test_split(
    X, y, w, test_size=0.2, stratify=y, random_state=42
)

# 8) XGB를 위한 레이블 인코딩
le = LabelEncoder().fit(y_tr)
y_tr_enc = le.transform(y_tr)
y_te_enc = le.transform(y_te)

# 9) 공통 CV 객체
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 10) Logistic Regression
grid_lr = GridSearchCV(
    LogisticRegression(max_iter=1000, multi_class='ovr', class_weight='balanced'),
    {'C':[0.01,0.1,1,10],'solver':['liblinear','saga']},
    cv=cv, scoring='accuracy', n_jobs=-1
)
grid_lr.fit(X_tr, y_tr, sample_weight=w_tr)
print("\nLogistic Best:", grid_lr.best_params_)
print("Logistic Acc:", accuracy_score(y_te, grid_lr.predict(X_te), sample_weight=w_te))

# 11) Random Forest
grid_rf = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42),
    {'n_estimators':[100,200],'max_depth':[None,10,20],'min_samples_leaf':[1,5]},
    cv=cv, scoring='accuracy', n_jobs=-1
)
grid_rf.fit(X_tr, y_tr, sample_weight=w_tr)
print("\nRF Best:", grid_rf.best_params_)
print("RF Acc:", accuracy_score(y_te, grid_rf.predict(X_te), sample_weight=w_te))

# 12) XGBoost
grid_xgb = GridSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    {'n_estimators':[100,200],'max_depth':[3,6],'learning_rate':[0.01,0.1]},
    cv=cv, scoring='accuracy', n_jobs=-1
)
grid_xgb.fit(X_tr, y_tr_enc, sample_weight=w_tr)
print("\nXGB Best:", grid_xgb.best_params_)
print("XGB Acc:", accuracy_score(y_te_enc, grid_xgb.predict(X_te), sample_weight=w_te))

# 13) SVM (RBF)
grid_svm = GridSearchCV(
    SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42),
    {'C':[0.1,1,10],'gamma':['scale','auto']},
    cv=cv, scoring='accuracy', n_jobs=-1
)
grid_svm.fit(X_tr, y_tr, sample_weight=w_tr)
print("\nSVM Best:", grid_svm.best_params_)
print("SVM Acc:", accuracy_score(y_te, grid_svm.predict(X_te), sample_weight=w_te))

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    top_k_accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import warnings

warnings.filterwarnings('ignore')

# 1) Load & melt
df = pd.read_csv("new_merged_data/df_result2_mapping1.csv", dtype=str)
age_cols = ['59이하','60-64','65-69','70-79','80-89','90이상']
m = df.melt(
    id_vars=['년도','구분','지역','상병코드','진료과'],
    value_vars=age_cols,
    var_name='age_group',
    value_name='count'
)
m['count'] = pd.to_numeric(m['count'], errors='coerce').fillna(0).astype(int)
m['대표진료과'] = m['진료과']
train = m[m['대표진료과'].notna()]

# 2) Prepare X, y, sample weights
train['year_num'] = train['년도'].astype(int) - train['년도'].astype(int).min()
X = train[['year_num','지역','age_group','구분']]
y = train['대표진료과']
w = train['count']

# 3) Train/test split
X_tr, X_te, y_tr, y_te, w_tr, w_te = train_test_split(
    X, y, w, test_size=0.2, stratify=y, random_state=42
)

# 4) Preprocessing
cat_cols = ['지역','age_group','구분']
num_cols = ['year_num']
preprocessor = ColumnTransformer([
    ('ohe', OneHotEncoder(drop='first', sparse_output=False), cat_cols),
    ('scale', StandardScaler(), num_cols),
])

# 5) Logistic Regression pipeline
pipe_lr = Pipeline([
    ('prep', preprocessor),
    ('select', SelectKBest(f_classif)),     # f_classif handles negative values
    ('clf', LogisticRegression(
        penalty='l1', solver='saga', max_iter=2000, class_weight='balanced'))
])
param_grid_lr = {
    'select__k': [30, 50, 100],
    'clf__C': [0.01, 0.1, 1],
}

# 6) XGBoost pipeline
pipe_xgb = Pipeline([
    ('prep', preprocessor),
    ('select', SelectKBest(f_classif)),
    ('clf', XGBClassifier(
        use_label_encoder=False, eval_metric='mlogloss', random_state=42))
])
param_grid_xgb = {
    'select__k': [30, 50, 100],
    'clf__n_estimators': [200, 400],
    'clf__max_depth': [3, 6],
    'clf__learning_rate': [0.01, 0.1],
    'clf__reg_alpha': [0, 1],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 7) Logistic GridSearchCV
grid_lr = GridSearchCV(
    pipe_lr, param_grid_lr, cv=cv,
    scoring='accuracy', n_jobs=-1, verbose=1
)
grid_lr.fit(X_tr, y_tr, clf__sample_weight=w_tr)
print("Logistic Best:", grid_lr.best_params_)
y_pred_lr = grid_lr.predict(X_te)
print("Logistic Acc:", accuracy_score(y_te, y_pred_lr, sample_weight=w_te))
print("Logistic Macro-F1:", f1_score(y_te, y_pred_lr, average='macro', sample_weight=w_te))

# 8) XGBoost GridSearchCV
# Label-encode y for XGBoost
le = LabelEncoder().fit(y_tr)
y_tr_enc = le.transform(y_tr)
y_te_enc = le.transform(y_te)

class XGBWrapper(XGBClassifier):
    def fit(self, X, y, **kwargs):
        return super().fit(X, y, sample_weight=kwargs.get('clf__sample_weight'))

# Replace the classifier in the pipeline
pipe_xgb.steps[-1] = ('clf', XGBWrapper(
    use_label_encoder=False, eval_metric='mlogloss', random_state=42
))

grid_xgb = GridSearchCV(
    pipe_xgb, param_grid_xgb, cv=cv,
    scoring='accuracy', n_jobs=-1, verbose=1
)
grid_xgb.fit(X_tr, y_tr_enc, clf__sample_weight=w_tr)
print("XGB Best:", grid_xgb.best_params_)
y_pred_xgb = grid_xgb.predict(X_te)
print("XGB Acc:", accuracy_score(y_te_enc, y_pred_xgb, sample_weight=w_te))
print("XGB Macro-F1:", f1_score(y_te_enc, y_pred_xgb, average='macro', sample_weight=w_te))

# 9) Detailed report for XGB
print("\nClassification Report (XGB):")
print(classification_report(
    y_te_enc, y_pred_xgb, sample_weight=w_te,
    target_names=le.classes_
))

# 8) XGBoost GridSearchCV … (생략) …
grid_xgb.fit(X_tr, y_tr_enc, clf__sample_weight=w_tr)
print("XGB Best:", grid_xgb.best_params_)
y_pred_xgb    = grid_xgb.predict(X_te)
y_proba_xgb   = grid_xgb.predict_proba(X_te)

# --- 기존 지표 ---
acc      = accuracy_score(y_te_enc, y_pred_xgb, sample_weight=w_te)
macro_f1 = f1_score(y_te_enc, y_pred_xgb, average='macro', sample_weight=w_te)
print("XGB Acc:", acc)
print("XGB Macro-F1:", macro_f1)

# --- 추가 평가 지표 ---
# 1) Top-3 Accuracy
top3 = top_k_accuracy_score(
    y_te_enc,
    y_proba_xgb,
    k=3,
    labels=np.arange(len(le.classes_)),
    sample_weight=w_te
)
print("XGB Top-3 Accuracy:", top3)

# 2) Balanced Accuracy
bal_acc = balanced_accuracy_score(y_te_enc, y_pred_xgb, sample_weight=w_te)
print("XGB Balanced Accuracy:", bal_acc)

# 3) Macro Precision & Recall
macro_prec = precision_score(
    y_te_enc, y_pred_xgb,
    average='macro', sample_weight=w_te
)
macro_rec  = recall_score(
    y_te_enc, y_pred_xgb,
    average='macro', sample_weight=w_te
)
print(f"XGB Macro-Precision: {macro_prec:.4f}")
print(f"XGB Macro-Recall:    {macro_rec:.4f}")

# 4) Macro ROC-AUC
#    이진화된 y_te_enc로 각 클래스별 ROC-AUC를 구해 평균
y_te_bin = label_binarize(y_te_enc, classes=np.arange(len(le.classes_)))
roc_auc = roc_auc_score(
    y_te_bin, y_proba_xgb,
    average='macro', sample_weight=w_te
)
print(f"XGB Macro ROC-AUC:   {roc_auc:.4f}")

# 5) 최종 Classification Report
print("\nClassification Report (XGB):")
print(classification_report(
    y_te_enc, y_pred_xgb,
    sample_weight=w_te,
    target_names=le.classes_
))
'''