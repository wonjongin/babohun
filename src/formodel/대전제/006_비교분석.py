import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

def interpret_cramers_v(cramers_v):
    """Cramer's V 해석"""
    if cramers_v < 0.1:
        return "약한 연관성"
    elif cramers_v < 0.3:
        return "중간 연관성"
    elif cramers_v < 0.5:
        return "강한 연관성"
    else:
        return "매우 강한 연관성"

def create_comparison_analysis():
    """두 분석 결과 비교"""
    
    print("=== 보훈병원 이용자 분포 비교 분석 ===")
    print("="*60)
    
    # 이전 분석 결과들
    # 1. 보훈병원 이용자 vs 보훈대상자
    bohun_comparison = {
        '연령대': 0.2014,
        '성별': 0.0670,
        '지역': 0.1029
    }
    
    # 2. 보훈병원 이용자 vs 전체 국민
    national_comparison = {
        '연령대': 0.1243,
        '성별': 0.0272,
        '지역': 0.0249
    }
    
    # 비교 테이블 생성
    comparison_data = []
    variables = ['연령대', '성별', '지역']
    
    for var in variables:
        bohun_v = bohun_comparison[var]
        national_v = national_comparison[var]
        difference = bohun_v - national_v
        
        comparison_data.append({
            '변수': var,
            '보훈대상자_비교': bohun_v,
            '전체국민_비교': national_v,
            '차이': difference,
            '보훈대상자_해석': interpret_cramers_v(bohun_v),
            '전체국민_해석': interpret_cramers_v(national_v)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # 결과 출력
    print("\n=== Cramer's V 비교 결과 ===")
    print("-" * 80)
    print(f"{'변수':<8} {'보훈대상자':<12} {'전체국민':<12} {'차이':<8} {'보훈대상자_해석':<12} {'전체국민_해석':<12}")
    print("-" * 80)
    
    for _, row in comparison_df.iterrows():
        print(f"{row['변수']:<8} {row['보훈대상자_비교']:<12.4f} {row['전체국민_비교']:<12.4f} "
              f"{row['차이']:<8.4f} {row['보훈대상자_해석']:<12} {row['전체국민_해석']:<12}")
    
    # 시각화
    create_comparison_visualization(comparison_df)
    
    # 해석 및 결론
    print_interpretation(comparison_df)
    
    return comparison_df

def create_comparison_visualization(comparison_df):
    """비교 결과 시각화"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    variables = comparison_df['변수'].values
    bohun_values = comparison_df['보훈대상자_비교'].values
    national_values = comparison_df['전체국민_비교'].values
    
    # 1. Cramer's V 직접 비교
    x = np.arange(len(variables))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, bohun_values, width, label='보훈대상자 비교', alpha=0.8, color='skyblue')
    bars2 = ax1.bar(x + width/2, national_values, width, label='전체국민 비교', alpha=0.8, color='lightcoral')
    
    ax1.set_xlabel('변수')
    ax1.set_ylabel("Cramer's V")
    ax1.set_title('변수별 연관성 강도 비교')
    ax1.set_xticks(x)
    ax1.set_xticklabels(variables)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 값 표시
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. 차이 분석
    differences = comparison_df['차이'].values
    colors = ['green' if d > 0 else 'red' for d in differences]
    
    bars3 = ax2.bar(variables, differences, color=colors, alpha=0.7)
    ax2.set_xlabel('변수')
    ax2.set_ylabel('차이 (보훈대상자 - 전체국민)')
    ax2.set_title('연관성 강도 차이 분석')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.grid(True, alpha=0.3)
    
    # 차이값 표시
    for bar, diff in zip(bars3, differences):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height > 0 else -0.02),
                f'{diff:.3f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    # 3. 연관성 강도 분포
    categories = ['약한 연관성', '중간 연관성', '강한 연관성', '매우 강한 연관성']
    bohun_counts = [sum(comparison_df['보훈대상자_해석'] == cat) for cat in categories]
    national_counts = [sum(comparison_df['전체국민_해석'] == cat) for cat in categories]
    
    x = np.arange(len(categories))
    bars4 = ax3.bar(x - width/2, bohun_counts, width, label='보훈대상자 비교', alpha=0.8, color='skyblue')
    bars5 = ax3.bar(x + width/2, national_counts, width, label='전체국민 비교', alpha=0.8, color='lightcoral')
    
    ax3.set_xlabel('연관성 강도')
    ax3.set_ylabel('변수 개수')
    ax3.set_title('연관성 강도 분포 비교')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 상대적 비율 분석
    relative_ratios = [b/n if n > 0 else 0 for b, n in zip(bohun_values, national_values)]
    
    bars6 = ax4.bar(variables, relative_ratios, color='orange', alpha=0.7)
    ax4.set_xlabel('변수')
    ax4.set_ylabel('상대적 비율 (보훈대상자/전체국민)')
    ax4.set_title('상대적 연관성 강도 비율')
    ax4.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='동일한 연관성')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # 비율값 표시
    for bar, ratio in zip(bars6, relative_ratios):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{ratio:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('src/formodel/대전제/비교분석_결과.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_interpretation(comparison_df):
    """해석 및 결론 출력"""
    
    print("\n" + "="*60)
    print("=== 분석 결과 해석 ===")
    print("="*60)
    
    # 1. 전반적 패턴 분석
    print("\n1. 전반적 패턴:")
    bohun_avg = comparison_df['보훈대상자_비교'].mean()
    national_avg = comparison_df['전체국민_비교'].mean()
    
    print(f"   • 보훈대상자 비교 평균 Cramer's V: {bohun_avg:.4f}")
    print(f"   • 전체국민 비교 평균 Cramer's V: {national_avg:.4f}")
    print(f"   • 평균 차이: {bohun_avg - national_avg:.4f}")
    
    if bohun_avg > national_avg:
        print("   → 보훈대상자와의 비교에서 더 강한 연관성을 보임")
    else:
        print("   → 전체국민과의 비교에서 더 강한 연관성을 보임")
    
    # 2. 변수별 분석
    print("\n2. 변수별 분석:")
    
    for _, row in comparison_df.iterrows():
        var = row['변수']
        bohun_v = row['보훈대상자_비교']
        national_v = row['전체국민_비교']
        diff = row['차이']
        
        print(f"\n   {var}:")
        print(f"   • 보훈대상자 비교: {bohun_v:.4f} ({row['보훈대상자_해석']})")
        print(f"   • 전체국민 비교: {national_v:.4f} ({row['전체국민_해석']})")
        print(f"   • 차이: {diff:.4f}")
        
        if diff > 0:
            print(f"   → 보훈대상자와의 비교에서 {abs(diff):.4f} 더 강한 연관성")
        else:
            print(f"   → 전체국민과의 비교에서 {abs(diff):.4f} 더 강한 연관성")
    
    # 3. 논리적 의미 해석
    print("\n3. 논리적 의미:")
    print("   • 보훈대상자 비교: 보훈 서비스 내부의 효율성과 특성 분석")
    print("   • 전체국민 비교: 보훈 서비스의 일반적 접근성과 대표성 분석")
    print("   • 차이가 클수록 보훈 서비스의 특수성이 강함을 의미")
    
    # 4. 실무적 시사점
    print("\n4. 실무적 시사점:")
    
    # 연령대 분석
    age_diff = comparison_df[comparison_df['변수'] == '연령대']['차이'].iloc[0]
    if age_diff > 0:
        print("   • 연령대: 보훈대상자 내에서 연령별 차이가 더 뚜렷함")
        print("     → 연령대별 맞춤형 보훈 서비스 제공이 중요")
    else:
        print("   • 연령대: 전체 국민과 유사한 연령 분포를 보임")
        print("     → 일반적인 의료 서비스와 유사한 접근 가능")
    
    # 성별 분석
    gender_diff = comparison_df[comparison_df['변수'] == '성별']['차이'].iloc[0]
    if gender_diff > 0:
        print("   • 성별: 보훈대상자 내에서 성별 차이가 더 뚜렷함")
        print("     → 성별별 차별화된 보훈 서비스 고려 필요")
    else:
        print("   • 성별: 전체 국민과 유사한 성별 분포를 보임")
        print("     → 성별 중립적인 서비스 제공 가능")
    
    # 지역 분석
    region_diff = comparison_df[comparison_df['변수'] == '지역']['차이'].iloc[0]
    if region_diff > 0:
        print("   • 지역: 보훈대상자 내에서 지역별 차이가 더 뚜렷함")
        print("     → 지역별 의료 접근성 개선이 시급함")
    else:
        print("   • 지역: 전체 국민과 유사한 지역 분포를 보임")
        print("     → 지역별 의료 서비스 접근성이 상대적으로 균등함")
    
    # 5. 결론
    print("\n5. 종합 결론:")
    positive_diffs = sum(comparison_df['차이'] > 0)
    total_vars = len(comparison_df)
    
    if positive_diffs > total_vars / 2:
        print("   • 보훈 서비스는 보훈대상자 집단 내에서 더 뚜렷한 특성을 보임")
        print("   • 보훈 서비스의 특수성과 차별화된 접근이 필요함")
    else:
        print("   • 보훈 서비스는 전체 국민과 유사한 특성을 보임")
        print("   • 일반적인 의료 서비스와 유사한 접근이 가능함")
    
    print("\n   → 보훈 서비스 정책 수립 시 이러한 특성을 고려해야 함")

def main():
    """메인 실행 함수"""
    
    print("=== 보훈병원 이용자 분포 비교 분석 ===")
    print("보훈대상자 비교 vs 전체국민 비교 결과 종합 분석")
    print("="*60)
    
    # 비교 분석 실행
    comparison_results = create_comparison_analysis()
    
    return comparison_results

if __name__ == "__main__":
    results = main() 