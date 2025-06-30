import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_model_results():
    """모델 결과 데이터 로드"""
    # 성능 비교 데이터
    comparison_df = pd.read_csv('src/formodel/진료과-진료비/v3_모델_성능_비교_상세.csv')
    
    # 시계열 피쳐 모델 결과
    timeseries_summary = pd.read_csv('model_results_new_진료과진료비_시계열/model_performance_summary.csv')
    timeseries_predictions = pd.read_csv('model_results_new_진료과진료비_시계열/prediction_results_detailed.csv')
    
    # 연령지역진료과 피쳐 모델 결과
    age_region_summary = pd.read_csv('model_results_new_진료과진료비_연령지역진료과/model_performance_summary.csv')
    age_region_predictions = pd.read_csv('model_results_new_진료과진료비_연령지역진료과/prediction_results_detailed.csv')
    
    return comparison_df, timeseries_summary, timeseries_predictions, age_region_summary, age_region_predictions

def analyze_model_performance():
    """모델 성능 종합 분석"""
    print("=" * 80)
    print("V3 모델 성능 종합 평가 리포트")
    print("=" * 80)
    
    # 데이터 로드
    comparison_df, timeseries_summary, timeseries_predictions, age_region_summary, age_region_predictions = load_model_results()
    
    # 1. 전체 성능 비교
    print("\n1. 전체 성능 비교")
    print("-" * 50)
    
    # 시계열 피쳐 모델 최고 성능
    timeseries_best = timeseries_summary.loc[timeseries_summary['Accuracy'].idxmax()]
    print(f"시계열 피쳐 모델 (Random Forest):")
    print(f"  - 정확도: {timeseries_best['Accuracy']:.3f}")
    print(f"  - F1 점수: {timeseries_best['F1_Score']:.3f}")
    print(f"  - CV 평균: {timeseries_best['CV_Mean']:.3f}")
    print(f"  - CV 표준편차: {timeseries_best['CV_Std']:.3f}")
    
    # 연령지역진료과 피쳐 모델 최고 성능
    age_region_best = age_region_summary.loc[age_region_summary['Accuracy'].idxmax()]
    print(f"\n연령지역진료과 피쳐 모델 (Logistic Regression):")
    print(f"  - 정확도: {age_region_best['Accuracy']:.3f}")
    print(f"  - F1 점수: {age_region_best['F1_Score']:.3f}")
    print(f"  - CV 평균: {age_region_best['CV_Mean']:.3f}")
    print(f"  - CV 표준편차: {age_region_best['CV_Std']:.3f}")
    
    # 성능 향상도 계산
    accuracy_improvement = (age_region_best['Accuracy'] - timeseries_best['Accuracy']) / timeseries_best['Accuracy'] * 100
    f1_improvement = (age_region_best['F1_Score'] - timeseries_best['F1_Score']) / timeseries_best['F1_Score'] * 100
    
    print(f"\n성능 향상도:")
    print(f"  - 정확도 향상: {accuracy_improvement:.1f}%")
    print(f"  - F1 점수 향상: {f1_improvement:.1f}%")
    
    # 2. 모델별 상세 성능 분석
    print("\n2. 모델별 상세 성능 분석")
    print("-" * 50)
    
    print("시계열 피쳐 모델들:")
    for _, row in timeseries_summary.iterrows():
        print(f"  {row['Model']}: 정확도={row['Accuracy']:.3f}, F1={row['F1_Score']:.3f}, CV={row['CV_Mean']:.3f}±{row['CV_Std']:.3f}")
    
    print("\n연령지역진료과 피쳐 모델들:")
    for _, row in age_region_summary.iterrows():
        print(f"  {row['Model']}: 정확도={row['Accuracy']:.3f}, F1={row['F1_Score']:.3f}, CV={row['CV_Mean']:.3f}±{row['CV_Std']:.3f}")
    
    # 3. 예측 정확도 분석
    print("\n3. 예측 정확도 분석")
    print("-" * 50)
    
    # 시계열 모델 예측 정확도
    timeseries_correct = timeseries_predictions['prediction_correct'].sum()
    timeseries_total = len(timeseries_predictions)
    timeseries_accuracy = timeseries_correct / timeseries_total
    
    # 연령지역진료과 모델 예측 정확도
    age_region_correct = age_region_predictions['prediction_correct'].sum()
    age_region_total = len(age_region_predictions)
    age_region_accuracy = age_region_correct / age_region_total
    
    print(f"시계열 피쳐 모델:")
    print(f"  - 정확한 예측: {timeseries_correct}/{timeseries_total} ({timeseries_accuracy:.3f})")
    
    print(f"\n연령지역진료과 피쳐 모델:")
    print(f"  - 정확한 예측: {age_region_correct}/{age_region_total} ({age_region_accuracy:.3f})")
    
    # 4. 클래스별 성능 분석
    print("\n4. 클래스별 성능 분석")
    print("-" * 50)
    
    # 시계열 모델 클래스별 분석
    timeseries_class_counts = timeseries_predictions['cost_bin_5'].value_counts().sort_index()
    timeseries_class_correct = timeseries_predictions.groupby('cost_bin_5')['prediction_correct'].sum()
    timeseries_class_accuracy = timeseries_class_correct / timeseries_class_counts
    
    print("시계열 피쳐 모델 클래스별 정확도:")
    for class_id in sorted(timeseries_class_accuracy.index):
        print(f"  클래스 {class_id}: {timeseries_class_accuracy[class_id]:.3f} ({timeseries_class_correct[class_id]}/{timeseries_class_counts[class_id]})")
    
    # 연령지역진료과 모델 클래스별 분석
    age_region_class_counts = age_region_predictions['cost_bin_5'].value_counts().sort_index()
    age_region_class_correct = age_region_predictions.groupby('cost_bin_5')['prediction_correct'].sum()
    age_region_class_accuracy = age_region_class_correct / age_region_class_counts
    
    print("\n연령지역진료과 피쳐 모델 클래스별 정확도:")
    for class_id in sorted(age_region_class_accuracy.index):
        print(f"  클래스 {class_id}: {age_region_class_accuracy[class_id]:.3f} ({age_region_class_correct[class_id]}/{age_region_class_counts[class_id]})")
    
    # 5. 진료과별 성능 분석
    print("\n5. 진료과별 성능 분석")
    print("-" * 50)
    
    # 시계열 모델 진료과별 분석
    timeseries_dept_accuracy = timeseries_predictions.groupby('진료과')['prediction_correct'].agg(['sum', 'count']).reset_index()
    timeseries_dept_accuracy['accuracy'] = timeseries_dept_accuracy['sum'] / timeseries_dept_accuracy['count']
    timeseries_dept_accuracy = timeseries_dept_accuracy.sort_values('accuracy', ascending=False)
    
    print("시계열 피쳐 모델 진료과별 정확도 (상위 10개):")
    for _, row in timeseries_dept_accuracy.head(10).iterrows():
        print(f"  {row['진료과']}: {row['accuracy']:.3f} ({row['sum']}/{row['count']})")
    
    # 연령지역진료과 모델 진료과별 분석
    age_region_dept_accuracy = age_region_predictions.groupby('진료과')['prediction_correct'].agg(['sum', 'count']).reset_index()
    age_region_dept_accuracy['accuracy'] = age_region_dept_accuracy['sum'] / age_region_dept_accuracy['count']
    age_region_dept_accuracy = age_region_dept_accuracy.sort_values('accuracy', ascending=False)
    
    print("\n연령지역진료과 피쳐 모델 진료과별 정확도 (상위 10개):")
    for _, row in age_region_dept_accuracy.head(10).iterrows():
        print(f"  {row['진료과']}: {row['accuracy']:.3f} ({row['sum']}/{row['count']})")
    
    # 6. 모델 안정성 분석
    print("\n6. 모델 안정성 분석")
    print("-" * 50)
    
    # CV 표준편차로 안정성 평가
    timeseries_stability = timeseries_summary['CV_Std'].mean()
    age_region_stability = age_region_summary['CV_Std'].mean()
    
    print(f"시계열 피쳐 모델 평균 CV 표준편차: {timeseries_stability:.4f}")
    print(f"연령지역진료과 피쳐 모델 평균 CV 표준편차: {age_region_stability:.4f}")
    
    if timeseries_stability < age_region_stability:
        print("시계열 피쳐 모델이 더 안정적입니다.")
    else:
        print("연령지역진료과 피쳐 모델이 더 안정적입니다.")
    
    # 7. 종합 평가 및 권장사항
    print("\n7. 종합 평가 및 권장사항")
    print("-" * 50)
    
    print("성능 비교 결과:")
    print(f"  - 정확도: 연령지역진료과 모델이 {accuracy_improvement:.1f}% 우수")
    print(f"  - F1 점수: 연령지역진료과 모델이 {f1_improvement:.1f}% 우수")
    print(f"  - 안정성: {'시계열' if timeseries_stability < age_region_stability else '연령지역진료과'} 모델이 더 안정적")
    
    print("\n권장사항:")
    if accuracy_improvement > 5 and f1_improvement > 10:
        print("  - 연령지역진료과 피쳐 모델을 주 모델로 사용 권장")
        print("  - 시계열 피쳐는 보조 피쳐로 활용 고려")
    elif abs(accuracy_improvement) < 5:
        print("  - 두 모델의 성능이 유사하므로 앙상블 방법 고려")
        print("  - 비즈니스 요구사항에 따라 선택")
    else:
        print("  - 시계열 피쳐 모델의 안정성을 고려하여 선택")
    
    return {
        'timeseries_best': timeseries_best,
        'age_region_best': age_region_best,
        'accuracy_improvement': accuracy_improvement,
        'f1_improvement': f1_improvement,
        'timeseries_class_accuracy': timeseries_class_accuracy,
        'age_region_class_accuracy': age_region_class_accuracy
    }

def create_performance_visualizations(results):
    """성능 시각화 생성"""
    # 1. 모델 성능 비교 차트
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 정확도 비교
    models = ['시계열 피쳐\n(Random Forest)', '연령지역진료과 피쳐\n(Logistic Regression)']
    accuracies = [results['timeseries_best']['Accuracy'], results['age_region_best']['Accuracy']]
    f1_scores = [results['timeseries_best']['F1_Score'], results['age_region_best']['F1_Score']]
    
    x = np.arange(len(models))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, accuracies, width, label='정확도', color='skyblue')
    axes[0, 0].bar(x + width/2, f1_scores, width, label='F1 점수', color='lightcoral')
    axes[0, 0].set_xlabel('모델')
    axes[0, 0].set_ylabel('점수')
    axes[0, 0].set_title('모델 성능 비교')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 성능 향상도
    improvements = [results['accuracy_improvement'], results['f1_improvement']]
    labels = ['정확도 향상', 'F1 점수 향상']
    colors = ['green' if x > 0 else 'red' for x in improvements]
    
    axes[0, 1].bar(labels, improvements, color=colors)
    axes[0, 1].set_ylabel('향상도 (%)')
    axes[0, 1].set_title('연령지역진료과 모델 대비 성능 향상도')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # 클래스별 정확도 비교
    classes = sorted(results['timeseries_class_accuracy'].index)
    timeseries_acc = [results['timeseries_class_accuracy'][c] for c in classes]
    age_region_acc = [results['age_region_class_accuracy'][c] for c in classes]
    
    x = np.arange(len(classes))
    axes[1, 0].bar(x - width/2, timeseries_acc, width, label='시계열 피쳐', color='skyblue')
    axes[1, 0].bar(x + width/2, age_region_acc, width, label='연령지역진료과 피쳐', color='lightcoral')
    axes[1, 0].set_xlabel('진료비 구간')
    axes[1, 0].set_ylabel('정확도')
    axes[1, 0].set_title('클래스별 정확도 비교')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([f'구간 {c}' for c in classes])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # CV 성능 비교
    cv_means = [results['timeseries_best']['CV_Mean'], results['age_region_best']['CV_Mean']]
    cv_stds = [results['timeseries_best']['CV_Std'], results['age_region_best']['CV_Std']]
    
    axes[1, 1].bar(models, cv_means, yerr=cv_stds, capsize=5, color=['skyblue', 'lightcoral'])
    axes[1, 1].set_ylabel('CV 평균 ± 표준편차')
    axes[1, 1].set_title('교차 검증 성능 비교')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('v3_모델_성능_종합_시각화.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n시각화 파일이 'v3_모델_성능_종합_시각화.png'로 저장되었습니다.")

if __name__ == "__main__":
    # 성능 분석 실행
    results = analyze_model_performance()
    
    # 시각화 생성
    create_performance_visualizations(results)
    
    print("\n" + "=" * 80)
    print("성능 평가 완료!")
    print("=" * 80) 