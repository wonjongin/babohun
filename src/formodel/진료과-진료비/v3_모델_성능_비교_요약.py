import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_and_compare_results():
    """두 모델의 결과를 로드하고 비교"""
    
    # 시계열 피쳐 추가 모델 결과
    timeseries_df = pd.read_csv('model_results_new_진료과진료비_시계열/model_performance_summary.csv')
    timeseries_df['Feature_Type'] = '시계열 피쳐'
    
    # 연령지역진료과 피쳐 추가 모델 결과
    age_region_df = pd.read_csv('model_results_new_진료과진료비_연령지역진료과/model_performance_summary.csv')
    age_region_df['Feature_Type'] = '연령지역진료과 피쳐'
    
    # 결과 병합
    combined_df = pd.concat([timeseries_df, age_region_df], ignore_index=True)
    
    return combined_df

def create_comparison_visualizations(df):
    """비교 시각화 생성"""
    
    # 1. 전체 성능 비교 (정확도 기준)
    plt.figure(figsize=(15, 10))
    
    # 서브플롯 1: 정확도 비교
    plt.subplot(2, 2, 1)
    for feature_type in df['Feature_Type'].unique():
        subset = df[df['Feature_Type'] == feature_type]
        plt.bar(subset['Model'], subset['Accuracy'], alpha=0.7, label=feature_type)
    
    plt.title('Model Accuracy Comparison by Feature Type')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 서브플롯 2: F1 점수 비교
    plt.subplot(2, 2, 2)
    for feature_type in df['Feature_Type'].unique():
        subset = df[df['Feature_Type'] == feature_type]
        plt.bar(subset['Model'], subset['F1_Score'], alpha=0.7, label=feature_type)
    
    plt.title('Model F1 Score Comparison by Feature Type')
    plt.ylabel('F1 Score')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 서브플롯 3: 교차 검증 성능 비교
    plt.subplot(2, 2, 3)
    for feature_type in df['Feature_Type'].unique():
        subset = df[df['Feature_Type'] == feature_type]
        plt.bar(subset['Model'], subset['CV_Mean'], alpha=0.7, label=feature_type)
    
    plt.title('Model CV Mean Comparison by Feature Type')
    plt.ylabel('CV Mean')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 서브플롯 4: 최고 성능 모델 비교
    plt.subplot(2, 2, 4)
    best_models = []
    for feature_type in df['Feature_Type'].unique():
        subset = df[df['Feature_Type'] == feature_type]
        best_model = subset.loc[subset['Accuracy'].idxmax()]
        best_models.append(best_model)
    
    best_df = pd.DataFrame(best_models)
    x = range(len(best_df))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], best_df['Accuracy'], width, label='Accuracy', alpha=0.8)
    plt.bar([i + width/2 for i in x], best_df['F1_Score'], width, label='F1 Score', alpha=0.8)
    
    plt.title('Best Model Performance by Feature Type')
    plt.ylabel('Score')
    plt.xticks(x, best_df['Feature_Type'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('v3_모델_성능_비교.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return best_df

def create_detailed_comparison_table(df):
    """상세 비교 테이블 생성"""
    
    # 각 피쳐 타입별 최고 성능 모델
    comparison_data = []
    
    for feature_type in df['Feature_Type'].unique():
        subset = df[df['Feature_Type'] == feature_type]
        
        # 정확도 기준 최고 모델
        best_accuracy = subset.loc[subset['Accuracy'].idxmax()]
        comparison_data.append({
            'Feature_Type': feature_type,
            'Best_Model_Accuracy': best_accuracy['Model'],
            'Best_Accuracy': best_accuracy['Accuracy'],
            'Best_Accuracy_F1': best_accuracy['F1_Score'],
            'Best_Accuracy_CV': best_accuracy['CV_Mean']
        })
        
        # F1 점수 기준 최고 모델
        best_f1 = subset.loc[subset['F1_Score'].idxmax()]
        comparison_data.append({
            'Feature_Type': feature_type,
            'Best_Model_F1': best_f1['Model'],
            'Best_F1_Score': best_f1['F1_Score'],
            'Best_F1_Accuracy': best_f1['Accuracy'],
            'Best_F1_CV': best_f1['CV_Mean']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv('v3_모델_성능_비교_상세.csv', index=False)
    
    return comparison_df

def main():
    """메인 함수"""
    print("=== v3 모델 성능 비교 분석 ===")
    
    # 결과 로드
    df = load_and_compare_results()
    
    print("\n=== 전체 결과 ===")
    print(df.to_string(index=False))
    
    # 시각화 생성
    print("\n=== 시각화 생성 중 ===")
    best_models = create_comparison_visualizations(df)
    
    # 상세 비교 테이블 생성
    print("\n=== 상세 비교 테이블 생성 중 ===")
    comparison_df = create_detailed_comparison_table(df)
    
    print("\n=== 최고 성능 모델 요약 ===")
    print(best_models[['Feature_Type', 'Model', 'Accuracy', 'F1_Score', 'CV_Mean']].to_string(index=False))
    
    # 성능 분석
    print("\n=== 성능 분석 ===")
    
    # 시계열 피쳐 모델 분석
    timeseries_best = df[df['Feature_Type'] == '시계열 피쳐'].loc[df[df['Feature_Type'] == '시계열 피쳐']['Accuracy'].idxmax()]
    print(f"시계열 피쳐 최고 성능 모델: {timeseries_best['Model']}")
    print(f"  - 정확도: {timeseries_best['Accuracy']:.4f}")
    print(f"  - F1 점수: {timeseries_best['F1_Score']:.4f}")
    print(f"  - CV 평균: {timeseries_best['CV_Mean']:.4f}")
    
    # 연령지역진료과 피쳐 모델 분석
    age_region_best = df[df['Feature_Type'] == '연령지역진료과 피쳐'].loc[df[df['Feature_Type'] == '연령지역진료과 피쳐']['Accuracy'].idxmax()]
    print(f"\n연령지역진료과 피쳐 최고 성능 모델: {age_region_best['Model']}")
    print(f"  - 정확도: {age_region_best['Accuracy']:.4f}")
    print(f"  - F1 점수: {age_region_best['F1_Score']:.4f}")
    print(f"  - CV 평균: {age_region_best['CV_Mean']:.4f}")
    
    # 성능 차이 분석
    accuracy_diff = age_region_best['Accuracy'] - timeseries_best['Accuracy']
    f1_diff = age_region_best['F1_Score'] - timeseries_best['F1_Score']
    
    print(f"\n=== 성능 차이 분석 ===")
    print(f"정확도 차이: {accuracy_diff:.4f} ({'연령지역진료과 피쳐가 우수' if accuracy_diff > 0 else '시계열 피쳐가 우수'})")
    print(f"F1 점수 차이: {f1_diff:.4f} ({'연령지역진료과 피쳐가 우수' if f1_diff > 0 else '시계열 피쳐가 우수'})")
    
    if accuracy_diff > 0:
        print(f"\n결론: 연령지역진료과 피쳐를 추가한 모델이 시계열 피쳐를 추가한 모델보다 {accuracy_diff:.4f} 높은 정확도를 보입니다.")
    else:
        print(f"\n결론: 시계열 피쳐를 추가한 모델이 연령지역진료과 피쳐를 추가한 모델보다 {abs(accuracy_diff):.4f} 높은 정확도를 보입니다.")

if __name__ == "__main__":
    main() 