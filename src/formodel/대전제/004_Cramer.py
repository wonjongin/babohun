import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

def clean_number_string(value):
    """숫자 문자열에서 쉼표 제거 및 정수 변환"""
    if pd.isna(value) or value == '-' or value == '':
        return 0
    
    value_str = str(value).strip()
    value_str = value_str.replace(',', '')
    value_str = re.sub(r'[^\d]', '', value_str)
    
    if value_str == '':
        return 0
    
    return int(value_str)

def calculate_cramers_v(contingency_table):
    """Cramer's V 계산"""
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
    
    # Cramer's V 계산
    n = contingency_table.sum()  # 전체 관측수
    min_dim = min(contingency_table.shape) - 1  # 최소 차원 - 1
    
    if min_dim == 0:
        cramers_v = 0
    else:
        cramers_v = np.sqrt(chi2_stat / (n * min_dim))
    
    return {
        'chi2_statistic': chi2_stat,
        'p_value': p_value,
        'degrees_of_freedom': dof,
        'cramers_v': cramers_v,
        'expected_frequencies': expected
    }

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

def analyze_age_distribution():
    """연령대별 분포 분석"""
    print("=== 연령대별 분포 분석 ===")
    
    # 데이터 로드
    hospital_data = pd.read_csv('new_merged_data/연령대별_총이용자수.csv')
    total_data = pd.read_csv('new_merged_data/보훈대상자_10세단위_전체인원.csv')
    
    # 100대 제외, 0~90대만 사용
    valid_ages = [f'{i}대' for i in range(0, 100, 10)]
    hosp_df = hospital_data[hospital_data['연령대'].isin(valid_ages)].copy()
    total_df = total_data[total_data['연령대'].isin(valid_ages)].copy()
    
    # 컬럼명 통일
    hosp_df = hosp_df.rename(columns={'총이용자수': '병원이용자수'})
    total_df = total_df.rename(columns={'인원수': '전체대상자수'})
    
    # 병합
    merged_df = pd.merge(hosp_df, total_df, on='연령대', how='inner')
    contingency_table = merged_df[['병원이용자수', '전체대상자수']].values
    
    # Cramer's V 계산
    results = calculate_cramers_v(contingency_table)
    
    print(f"카이제곱 통계량: {results['chi2_statistic']:.4f}")
    print(f"p-value: {results['p_value']:.6f}")
    print(f"Cramer's V: {results['cramers_v']:.4f}")
    print(f"연관성 강도: {interpret_cramers_v(results['cramers_v'])}")
    
    return results, merged_df

def analyze_gender_distribution():
    """성별 분포 분석"""
    print("\n=== 성별 분포 분석 ===")
    
    # 데이터 로드
    hospital_data = pd.read_csv('new_merged_data/전체_성별_이용자수.csv')
    total_data = pd.read_csv('new_merged_data/국가보훈부_보훈대상자 성별연령별 실인원현황.csv')
    
    # 2024-12-31 기준일만 필터링
    total_data = total_data[total_data['기준일'] == '2024-12-31'].copy()
    
    # 숫자 데이터 정리
    total_data['남'] = total_data['남'].apply(clean_number_string)
    total_data['여'] = total_data['여'].apply(clean_number_string)
    
    # 성별별 총합 계산
    total_male = total_data['남'].sum()
    total_female = total_data['여'].sum()
    
    total_gender_df = pd.DataFrame({
        '성별': ['남', '여'],
        '전체대상자수': [total_male, total_female]
    })
    
    # 병원 이용자 데이터 컬럼명 통일
    hospital_data = hospital_data.rename(columns={'이용자수': '병원이용자수'})
    
    # 병합
    merged_df = pd.merge(hospital_data, total_gender_df, on='성별', how='inner')
    contingency_table = merged_df[['병원이용자수', '전체대상자수']].values
    
    # Cramer's V 계산
    results = calculate_cramers_v(contingency_table)
    
    print(f"카이제곱 통계량: {results['chi2_statistic']:.4f}")
    print(f"p-value: {results['p_value']:.6f}")
    print(f"Cramer's V: {results['cramers_v']:.4f}")
    print(f"연관성 강도: {interpret_cramers_v(results['cramers_v'])}")
    
    return results, merged_df

def analyze_region_distribution():
    """지역별 분포 분석"""
    print("\n=== 지역별 분포 분석 ===")
    
    # 데이터 로드
    hospital_data = pd.read_csv('new_merged_data/지역별_총이용자수.csv')
    total_data = pd.read_csv('new_merged_data/보훈대상자_권역별_총인원.csv')
    
    # 지역명 매핑
    region_mapping = {
        '광주권역': '광주',
        '대구권역': '대구', 
        '대전권역': '대전',
        '부산권역': '부산',
        '인천권역': '인천',
        '중앙권역': '중앙'
    }
    
    # 전체 대상자 데이터의 지역명 변경
    total_data['지역'] = total_data['권역'].map(region_mapping)
    total_data = total_data.drop('권역', axis=1)
    
    # 컬럼명 통일
    hospital_data = hospital_data.rename(columns={'총이용자수': '병원이용자수'})
    total_data = total_data.rename(columns={'보훈대상자수': '전체대상자수'})
    
    # 병합
    merged_df = pd.merge(hospital_data, total_data, on='지역', how='inner')
    contingency_table = merged_df[['병원이용자수', '전체대상자수']].values
    
    # Cramer's V 계산
    results = calculate_cramers_v(contingency_table)
    
    print(f"카이제곱 통계량: {results['chi2_statistic']:.4f}")
    print(f"p-value: {results['p_value']:.6f}")
    print(f"Cramer's V: {results['cramers_v']:.4f}")
    print(f"연관성 강도: {interpret_cramers_v(results['cramers_v'])}")
    
    return results, merged_df

def create_summary_visualization(age_results, gender_results, region_results):
    """결과 요약 시각화"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Cramer's V 비교
    variables = ['연령대', '성별', '지역']
    cramers_v_values = [
        age_results['cramers_v'],
        gender_results['cramers_v'],
        region_results['cramers_v']
    ]
    
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    bars = ax1.bar(variables, cramers_v_values, color=colors, alpha=0.8)
    
    # 값 표시
    for bar, value in zip(bars, cramers_v_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_ylabel("Cramer's V")
    ax1.set_title("변수별 연관성 강도 (Cramer's V)")
    ax1.set_ylim(0, max(cramers_v_values) * 1.2)
    ax1.grid(True, alpha=0.3)
    
    # 2. 연관성 강도 해석
    interpretations = [
        interpret_cramers_v(age_results['cramers_v']),
        interpret_cramers_v(gender_results['cramers_v']),
        interpret_cramers_v(region_results['cramers_v'])
    ]
    
    # 색상 매핑
    color_map = {
        '약한 연관성': 'lightgray',
        '중간 연관성': 'lightblue',
        '강한 연관성': 'orange',
        '매우 강한 연관성': 'red'
    }
    
    colors = [color_map[interpretation] for interpretation in interpretations]
    
    bars2 = ax2.bar(variables, [1, 1, 1], color=colors, alpha=0.8)
    ax2.set_ylabel("연관성 강도")
    ax2.set_title("변수별 연관성 강도 해석")
    ax2.set_ylim(0, 1.2)
    
    # 범례 추가
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.8, label=label) 
                      for label, color in color_map.items()]
    ax2.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('src/formodel/대전제/Cramers_V_분석_결과.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """메인 실행 함수"""
    
    print("=== 보훈병원 이용자 vs 전체 보훈대상자 연관성 분석 (Cramer's V) ===\n")
    
    # 1. 연령대별 분석
    age_results, age_data = analyze_age_distribution()
    
    # 2. 성별 분석
    gender_results, gender_data = analyze_gender_distribution()
    
    # 3. 지역별 분석
    region_results, region_data = analyze_region_distribution()
    
    # 4. 종합 결과
    print("\n" + "="*60)
    print("=== 종합 분석 결과 ===")
    print("="*60)
    
    results_summary = [
        ("연령대", age_results['cramers_v'], age_results['p_value']),
        ("성별", gender_results['cramers_v'], gender_results['p_value']),
        ("지역", region_results['cramers_v'], region_results['p_value'])
    ]
    
    print(f"{'변수':<10} {'Cramer\'s V':<12} {'p-value':<12} {'연관성 강도':<15}")
    print("-" * 60)
    
    for variable, cramers_v, p_value in results_summary:
        interpretation = interpret_cramers_v(cramers_v)
        print(f"{variable:<10} {cramers_v:<12.4f} {p_value:<12.6f} {interpretation:<15}")
    
    # 5. 시각화
    print("\n5. 결과 시각화 중...")
    create_summary_visualization(age_results, gender_results, region_results)
    
    # 6. 해석 가이드
    print("\n" + "="*60)
    print("=== Cramer's V 해석 가이드 ===")
    print("="*60)
    print("• 0.1 미만: 약한 연관성")
    print("• 0.1 ~ 0.3: 중간 연관성")
    print("• 0.3 ~ 0.5: 강한 연관성")
    print("• 0.5 이상: 매우 강한 연관성")
    print("\n※ Cramer's V는 0~1 사이의 값을 가지며, 값이 클수록 연관성이 강함")
    
    return {
        'age': age_results,
        'gender': gender_results,
        'region': region_results
    }

if __name__ == "__main__":
    results = main()
