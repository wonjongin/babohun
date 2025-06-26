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

def analyze_age_distribution_vs_national():
    """연령대별 분포 분석 (보훈병원 이용자 vs 전체 국민)"""
    print("=== 연령대별 분포 분석 (보훈병원 이용자 vs 전체 국민) ===")
    
    # 1. 보훈병원 이용자 데이터 로드
    hospital_data = pd.read_csv('new_merged_data/연령대별_총이용자수.csv')
    
    # 2. 우리나라 전체 인구 데이터 로드
    national_data = pd.read_csv('mpva_original_data/2024우리나라인구.csv')
    
    # 전국 데이터만 추출
    national_total = national_data[national_data['행정구역'].str.contains('전국')].iloc[0]
    
    # 연령대별 인구수 추출 (0~9세부터 90~99세까지)
    age_columns = ['2024년_계_0~9세', '2024년_계_10~19세', '2024년_계_20~29세', 
                   '2024년_계_30~39세', '2024년_계_40~49세', '2024년_계_50~59세',
                   '2024년_계_60~69세', '2024년_계_70~79세', '2024년_계_80~89세', 
                   '2024년_계_90~99세']
    
    age_labels = ['0대', '10대', '20대', '30대', '40대', '50대', '60대', '70대', '80대', '90대']
    
    national_age_data = []
    for col, label in zip(age_columns, age_labels):
        population = clean_number_string(national_total[col])
        national_age_data.append({'연령대': label, '전체국민수': population})
    
    national_df = pd.DataFrame(national_age_data)
    
    # 3. 보훈병원 이용자 데이터 전처리 (0~90대만)
    valid_ages = age_labels
    hosp_df = hospital_data[hospital_data['연령대'].isin(valid_ages)].copy()
    hosp_df = hosp_df.rename(columns={'총이용자수': '병원이용자수'})
    
    # 4. 병합
    merged_df = pd.merge(hosp_df, national_df, on='연령대', how='inner')
    contingency_table = merged_df[['병원이용자수', '전체국민수']].values
    
    # 5. Cramer's V 계산
    results = calculate_cramers_v(contingency_table)
    
    print(f"카이제곱 통계량: {results['chi2_statistic']:.4f}")
    print(f"p-value: {results['p_value']:.6f}")
    print(f"Cramer's V: {results['cramers_v']:.4f}")
    print(f"연관성 강도: {interpret_cramers_v(results['cramers_v'])}")
    
    return results, merged_df

def analyze_gender_distribution_vs_national():
    """성별 분포 분석 (보훈병원 이용자 vs 전체 국민)"""
    print("\n=== 성별 분포 분석 (보훈병원 이용자 vs 전체 국민) ===")
    
    # 1. 보훈병원 이용자 데이터 로드
    hospital_data = pd.read_csv('new_merged_data/전체_성별_이용자수.csv')
    
    # 2. 우리나라 전체 인구 데이터 로드
    national_data = pd.read_csv('mpva_original_data/2024우리나라인구.csv')
    
    # 전국 데이터만 추출
    national_total = national_data[national_data['행정구역'].str.contains('전국')].iloc[0]
    
    # 성별 인구수 추출
    male_population = clean_number_string(national_total['2024년_남_총인구수'])
    female_population = clean_number_string(national_total['2024년_여_총인구수'])
    
    national_gender_df = pd.DataFrame({
        '성별': ['남', '여'],
        '전체국민수': [male_population, female_population]
    })
    
    # 3. 병원 이용자 데이터 컬럼명 통일
    hospital_data = hospital_data.rename(columns={'이용자수': '병원이용자수'})
    
    # 4. 병합
    merged_df = pd.merge(hospital_data, national_gender_df, on='성별', how='inner')
    contingency_table = merged_df[['병원이용자수', '전체국민수']].values
    
    # 5. Cramer's V 계산
    results = calculate_cramers_v(contingency_table)
    
    print(f"카이제곱 통계량: {results['chi2_statistic']:.4f}")
    print(f"p-value: {results['p_value']:.6f}")
    print(f"Cramer's V: {results['cramers_v']:.4f}")
    print(f"연관성 강도: {interpret_cramers_v(results['cramers_v'])}")
    
    return results, merged_df

def analyze_region_distribution_vs_national():
    """지역별 분포 분석 (보훈병원 이용자 vs 전체 국민)"""
    print("\n=== 지역별 분포 분석 (보훈병원 이용자 vs 전체 국민) ===")
    
    # 1. 보훈병원 이용자 데이터 로드
    hospital_data = pd.read_csv('new_merged_data/지역별_총이용자수.csv')
    
    # 2. 우리나라 전체 인구 데이터 로드
    national_data = pd.read_csv('mpva_original_data/2024우리나라인구.csv')
    
    # 지역별 인구수 추출 (보훈병원 권역과 올바르게 매칭)
    region_mapping = {
        '서울특별시': '중앙',
        '경기도': '중앙',
        '부산광역시': '부산',
        '울산광역시': '부산',
        '경상남도': '부산',
        '대구광역시': '대구',
        '경상북도': '대구',
        '광주광역시': '광주',
        '전라남도': '광주',
        '전북특별자치도': '광주',
        '대전광역시': '대전',
        '세종특별자치시': '대전',
        '충청북도': '대전',
        '충청남도': '대전',
        '인천광역시': '인천'
    }
    
    national_region_data = []
    for _, row in national_data.iterrows():
        region_name = row['행정구역'].split()[0]  # 첫 번째 단어만 추출
        if region_name in region_mapping:
            mapped_region = region_mapping[region_name]
            population = clean_number_string(row['2024년_계_총인구수'])
            national_region_data.append({
                '지역': mapped_region,
                '전체국민수': population
            })
    
    # 권역별로 인구수 합계 계산
    national_region_df = pd.DataFrame(national_region_data)
    national_region_df = national_region_df.groupby('지역')['전체국민수'].sum().reset_index()
    
    # 3. 병원 이용자 데이터 컬럼명 통일
    hospital_data = hospital_data.rename(columns={'총이용자수': '병원이용자수'})
    
    # 4. 병합
    merged_df = pd.merge(hospital_data, national_region_df, on='지역', how='inner')
    contingency_table = merged_df[['병원이용자수', '전체국민수']].values
    
    # 5. Cramer's V 계산
    results = calculate_cramers_v(contingency_table)
    
    print(f"카이제곱 통계량: {results['chi2_statistic']:.4f}")
    print(f"p-value: {results['p_value']:.6f}")
    print(f"Cramer's V: {results['cramers_v']:.4f}")
    print(f"연관성 강도: {interpret_cramers_v(results['cramers_v'])}")
    
    # 6. 권역별 인구수 확인 출력
    print("\n권역별 전체국민 인구수:")
    for _, row in national_region_df.iterrows():
        print(f"  {row['지역']}: {row['전체국민수']:,}명")
    
    return results, merged_df

def create_comparison_visualization(age_results, gender_results, region_results):
    """결과 비교 시각화"""
    
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
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_ylabel("Cramer's V")
    ax1.set_title("보훈병원 이용자 vs 전체 국민 연관성 강도")
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
    plt.savefig('src/formodel/대전제/Cramers_V_국민비교_결과.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """메인 실행 함수"""
    
    print("=== 보훈병원 이용자 vs 전체 국민 연관성 분석 (Cramer's V) ===\n")
    
    # 1. 연령대별 분석
    age_results, age_data = analyze_age_distribution_vs_national()
    
    # 2. 성별 분석
    gender_results, gender_data = analyze_gender_distribution_vs_national()
    
    # 3. 지역별 분석
    region_results, region_data = analyze_region_distribution_vs_national()
    
    # 4. 종합 결과
    print("\n" + "="*60)
    print("=== 종합 분석 결과 (보훈병원 이용자 vs 전체 국민) ===")
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
    create_comparison_visualization(age_results, gender_results, region_results)
    
    # 6. 해석 가이드
    print("\n" + "="*60)
    print("=== Cramer's V 해석 가이드 ===")
    print("="*60)
    print("• 0.1 미만: 약한 연관성")
    print("• 0.1 ~ 0.3: 중간 연관성")
    print("• 0.3 ~ 0.5: 강한 연관성")
    print("• 0.5 이상: 매우 강한 연관성")
    print("\n※ Cramer's V는 0~1 사이의 값을 가지며, 값이 클수록 연관성이 강함")
    
    # 7. 논리적 비교 설명
    print("\n" + "="*60)
    print("=== 논리적 비교의 의미 ===")
    print("="*60)
    print("• 보훈병원 이용자 vs 전체 국민: 보훈병원 서비스의 일반적 접근성 분석")
    print("• 보훈병원 이용자 vs 보훈대상자: 보훈 서비스 내부의 효율성 분석")
    print("• 두 분석을 비교하여 보훈 서비스의 특성과 일반 의료 서비스와의 차이점 파악")
    
    return {
        'age': age_results,
        'gender': gender_results,
        'region': region_results
    }

if __name__ == "__main__":
    results = main() 