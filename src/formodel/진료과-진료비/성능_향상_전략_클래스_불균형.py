import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """데이터 로드 및 전처리"""
    # 기존 데이터 로드
    data = pd.read_csv('model_results_new_진료과진료비_연령지역진료과/prediction_results_detailed.csv')
    
    # cost_bin_5 NaN 제거
    data = data[~data['cost_bin_5'].isna()]
    
    # 피쳐와 타겟 분리
    feature_columns = [col for col in data.columns if col not in ['cost_bin_5', '진료비(천원)', 'y_actual', 'y_predicted', 'prediction_correct']]
    X = data[feature_columns]
    y = data['cost_bin_5']
    
    # 훈련/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test, feature_columns

def analyze_class_distribution(y_train, y_test):
    """클래스 분포 분석"""
    print("=" * 60)
    print("클래스 분포 분석")
    print("=" * 60)
    
    print("\n훈련 데이터 클래스 분포:")
    train_dist = y_train.value_counts().sort_index()
    for class_id, count in train_dist.items():
        percentage = (count / len(y_train)) * 100
        print(f"  클래스 {class_id}: {count}개 ({percentage:.1f}%)")
    
    print("\n테스트 데이터 클래스 분포:")
    test_dist = y_test.value_counts().sort_index()
    for class_id, count in test_dist.items():
        percentage = (count / len(y_test)) * 100
        print(f"  클래스 {class_id}: {count}개 ({percentage:.1f}%)")

def apply_sampling_methods(X_train, y_train):
    """다양한 샘플링 방법 적용"""
    sampling_methods = {
        'Original': (X_train, y_train),
        'SMOTE': SMOTE(random_state=42),
        'ADASYN': ADASYN(random_state=42),
        'BorderlineSMOTE': BorderlineSMOTE(random_state=42),
        'RandomUnderSampler': RandomUnderSampler(random_state=42),
        'TomekLinks': TomekLinks(),
        'SMOTEENN': SMOTEENN(random_state=42),
        'SMOTETomek': SMOTETomek(random_state=42)
    }
    
    results = {}
    
    for method_name, sampler in sampling_methods.items():
        print(f"\n{method_name} 적용 중...")
        
        if method_name == 'Original':
            X_resampled, y_resampled = X_train, y_train
        else:
            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        
        # 분포 확인
        dist = pd.Series(y_resampled).value_counts().sort_index()
        print(f"  재샘플링 후 분포:")
        for class_id, count in dist.items():
            percentage = (count / len(y_resampled)) * 100
            print(f"    클래스 {class_id}: {count}개 ({percentage:.1f}%)")
        
        results[method_name] = (X_resampled, y_resampled)
    
    return results

def evaluate_sampling_methods(sampling_results, X_test, y_test):
    """샘플링 방법별 성능 평가"""
    print("\n" + "=" * 60)
    print("샘플링 방법별 성능 평가")
    print("=" * 60)
    
    evaluation_results = []
    
    for method_name, (X_resampled, y_resampled) in sampling_results.items():
        print(f"\n{method_name} 평가 중...")
        
        # 모델 훈련 (Random Forest 사용)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_resampled, y_resampled)
        
        # 예측
        y_pred = model.predict(X_test)
        
        # 성능 평가
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        print(f"  정확도: {accuracy:.3f}")
        print(f"  F1 (Macro): {f1_macro:.3f}")
        print(f"  F1 (Weighted): {f1_weighted:.3f}")
        
        evaluation_results.append({
            'Method': method_name,
            'Accuracy': accuracy,
            'F1_Macro': f1_macro,
            'F1_Weighted': f1_weighted
        })
    
    return pd.DataFrame(evaluation_results)

def apply_class_weights(X_train, y_train, X_test, y_test):
    """클래스 가중치 적용"""
    print("\n" + "=" * 60)
    print("클래스 가중치 적용")
    print("=" * 60)
    
    # 클래스별 가중치 계산
    class_counts = y_train.value_counts()
    total_samples = len(y_train)
    class_weights = {class_id: total_samples / (len(class_counts) * count) 
                    for class_id, count in class_counts.items()}
    
    print("클래스별 가중치:")
    for class_id, weight in class_weights.items():
        print(f"  클래스 {class_id}: {weight:.3f}")
    
    # 가중치를 적용한 모델들
    models = {
        'RandomForest (Weighted)': RandomForestClassifier(
            n_estimators=100, 
            class_weight=class_weights, 
            random_state=42
        ),
        'LogisticRegression (Weighted)': LogisticRegression(
            class_weight=class_weights, 
            random_state=42,
            max_iter=1000
        )
    }
    
    results = []
    
    for model_name, model in models.items():
        print(f"\n{model_name} 평가 중...")
        
        # 모델 훈련
        model.fit(X_train, y_train)
        
        # 예측
        y_pred = model.predict(X_test)
        
        # 성능 평가
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        print(f"  정확도: {accuracy:.3f}")
        print(f"  F1 (Macro): {f1_macro:.3f}")
        print(f"  F1 (Weighted): {f1_weighted:.3f}")
        
        results.append({
            'Method': model_name,
            'Accuracy': accuracy,
            'F1_Macro': f1_macro,
            'F1_Weighted': f1_weighted
        })
    
    return pd.DataFrame(results)

def main():
    """메인 실행 함수"""
    print("클래스 불균형 문제 해결을 위한 성능 향상 전략")
    print("=" * 60)
    
    # 데이터 로드
    X_train, X_test, y_train, y_test, feature_columns = load_and_prepare_data()
    
    # 클래스 분포 분석
    analyze_class_distribution(y_train, y_test)
    
    # 샘플링 방법 적용
    sampling_results = apply_sampling_methods(X_train, y_train)
    
    # 샘플링 방법별 성능 평가
    sampling_evaluation = evaluate_sampling_methods(sampling_results, X_test, y_test)
    
    # 클래스 가중치 적용
    weight_evaluation = apply_class_weights(X_train, y_train, X_test, y_test)
    
    # 결과 비교
    print("\n" + "=" * 60)
    print("최종 성능 비교")
    print("=" * 60)
    
    all_results = pd.concat([sampling_evaluation, weight_evaluation], ignore_index=True)
    all_results = all_results.sort_values('F1_Macro', ascending=False)
    
    print("\nF1 (Macro) 기준 상위 5개 방법:")
    print(all_results.head().to_string(index=False))
    
    # 최고 성능 방법 저장
    best_method = all_results.iloc[0]
    print(f"\n최고 성능 방법: {best_method['Method']}")
    print(f"  정확도: {best_method['Accuracy']:.3f}")
    print(f"  F1 (Macro): {best_method['F1_Macro']:.3f}")
    print(f"  F1 (Weighted): {best_method['F1_Weighted']:.3f}")
    
    # 결과 저장
    all_results.to_csv('클래스_불균형_해결_결과.csv', index=False)
    print(f"\n결과가 '클래스_불균형_해결_결과.csv'에 저장되었습니다.")

if __name__ == "__main__":
    main() 