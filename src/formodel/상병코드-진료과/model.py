import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from numpy import interp

import matplotlib
matplotlib.rc('font', family='Pretendard')
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지


# 데이터 불러오기
df = pd.read_csv('new_merged_data/상병코드별_진료과_건수_더미_병합.csv')

# 정답(y) 만들기: 각 행에서 건수가 가장 큰 진료과를 정답으로
X = df.drop('상명코드', axis=1)
y = X.idxmax(axis=1)  # 각 행에서 가장 큰 값의 컬럼명(진료과)

# train/test 분리
le = LabelEncoder()
y_num = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_num, test_size=0.2, random_state=42)

# 모델 리스트
models = {
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    'LightGBM': LGBMClassifier(),
    'RandomForest': RandomForestClassifier(),
    'CatBoost': CatBoostClassifier(verbose=0),
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'SVM': SVC(probability=True),
}

# y를 one-hot 인코딩
classes = np.unique(y_num)
y_test_bin = label_binarize(y_test, classes=classes)
n_classes = len(classes)

plt.figure(figsize=(12, 10))
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    # micro-average
    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (micro-AUC={roc_auc:.2f})')

    # confusion matrix 등은 별도 figure로
    labels_in_test = np.unique(y_test)
    target_names_in_test = le.inverse_transform(labels_in_test)
    print(f"{name} classification report:\n", classification_report(y_test, y_pred, labels=labels_in_test, target_names=target_names_in_test))
    cm = confusion_matrix(y_test, y_pred, labels=labels_in_test)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names_in_test, yticklabels=target_names_in_test)
    plt.xlabel('예측 진료과')
    plt.ylabel('실제 진료과')
    plt.title(f'Confusion Matrix - {name}')
    plt.tight_layout()
    plt.show()

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('모델별 ROC 곡선 (micro-average)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 상병코드로 진료과 예측 함수
def predict_department_by_code(code, model, df):
    row = df[df['상명코드'] == code]
    if row.empty:
        return "해당 상병코드 없음"
    X_input = row.drop('상명코드', axis=1)
    pred_num = model.predict(X_input)
    pred = le.inverse_transform(pred_num)
    return pred[0]

def predict_department_by_code_all_models(code, models, df, label_encoder=None):
    row = df[df['상명코드'] == code]
    if row.empty:
        return "해당 상병코드 없음"
    X_input = row.drop('상명코드', axis=1)
    results = {}
    for name, model in models.items():
        if label_encoder:
            pred_num = model.predict(X_input)
            pred = label_encoder.inverse_transform(pred_num)
            results[name] = pred[0]
        else:
            pred = model.predict(X_input)
            results[name] = pred[0]
    return results

# 사용 예시
code = input("상병코드를 입력하세요: ")
results = predict_department_by_code_all_models(code, models, df, le)
for name, pred in results.items():
    print(f"{name} → 예측 진료과: {pred}")

print("all_fpr:", all_fpr)
print("mean_tpr:", mean_tpr)
